"""
ZImage Unified Model for DMD2 Training

Wraps the generator (feedforward_model) and the guidance model.

Architecture:
  ZImageUniModel
  ├── feedforward_model: ZImageTransformer2DModel  (generator G_θ, trainable)
  ├── guidance_model: ZImageGuidance
  │   ├── real_transformer: ZImageTransformer2DModel  (frozen, models real dist)
  │   └── fake_transformer: ZImageTransformer2DModel  (trainable, models fake dist)
  ├── text_encoder: LLM-based encoder (e.g. Qwen)
  ├── tokenizer: AutoTokenizer with chat template
  └── vae: AutoencoderKL (16-channel)

Generator forward (1-step synthesis):
  noise → feedforward_model(noise, t_gen, cap_feats) → x0_pred
  x0_pred = noise + sigma_gen * raw_output   (flow matching)

Key conventions:
- t_norm = (1000 - T) / 1000, where 0=noisy, 1=clean
- sigma_gen: noise level at the generation conditioning timestep (close to 1.0)
- VAE decode: latents → (latents / scaling_factor) + shift_factor → vae.decode
"""
from diffusers import AutoencoderKL
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from transformers import AutoTokenizer
from main.utils import NoOpContext
from main.zimage.zimage_guidance import ZImageGuidance, latents_to_list, list_to_latents
from torch import nn
import torch


def encode_prompts(text_encoder, tokenizer, prompts: list, device, max_length: int = 512):
    """
    Encode a list of text prompts using ZImage's chat-template tokenization.

    Replicates ZImagePipeline._encode_prompt logic for standalone use.

    Returns:
        cap_feats_list: list of [seq_len_i, cap_feat_dim] tensors (non-padded tokens only)
    """
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    text_inputs = tokenizer(
        formatted,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device).bool()

    with torch.no_grad():
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-2]  # second-to-last hidden state

    # Return variable-length embeddings (masked to non-padding tokens)
    return [hidden[i][attention_mask[i]] for i in range(len(prompts))]


class ZImageUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.accelerator = accelerator
        self.use_fp16 = args.use_fp16
        self.gradient_checkpointing = args.gradient_checkpointing

        # ── Guidance model (real + fake transformers) ────────────────────────
        self.guidance_model = ZImageGuidance(args, accelerator)
        self.num_train_timesteps = self.guidance_model.num_train_timesteps

        # ── Generator (feedforward) transformer ─────────────────────────────
        # Load in bfloat16 when use_fp16 is set: ZImage is 6B params and
        # bf16 keeps gradient precision adequate while halving memory.
        gen_dtype = torch.bfloat16 if args.use_fp16 else torch.float32
        self.feedforward_model = ZImageTransformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=gen_dtype
        )
        # Homogenize all parameter dtypes (RMSNorm weights can stay float32 after
        # from_pretrained even with torch_dtype=bfloat16).  FSDP requires every
        # tensor within one FSDP unit to share a single dtype.
        self.feedforward_model.to(gen_dtype)
        self.feedforward_model.requires_grad_(True)

        if self.gradient_checkpointing:
            self.feedforward_model.enable_gradient_checkpointing()

        # ── Sigma for generator conditioning ────────────────────────────────
        # conditioning_sigma_index: index into precomputed sigmas; use the
        # largest sigma (index 0, closest to pure noise) for 1-step generation.
        sigma_idx = args.conditioning_sigma_index
        self.conditioning_sigma = self.guidance_model.sigmas[sigma_idx].item()
        self.conditioning_t_norm = self.guidance_model.t_norms[sigma_idx].item()

        # ── Text encoder and tokenizer ───────────────────────────────────────
        # ZImage uses a Qwen-based causal LM with chat template + thinking.
        # AutoModelForCausalLM reads the subfolder's config.json to pick the right class.
        from transformers import AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        text_enc_dtype = torch.bfloat16 if args.use_fp16 else torch.float32
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            args.model_id, subfolder="text_encoder", dtype=text_enc_dtype
        ).to(accelerator.device)
        self.text_encoder.requires_grad_(False)

        # ── VAE ──────────────────────────────────────────────────────────────
        self.vae = AutoencoderKL.from_pretrained(
            args.model_id, subfolder="vae"
        ).float().to(accelerator.device)
        self.vae.requires_grad_(False)
        if self.use_fp16:
            self.vae.to(torch.float16)

        self.num_visuals = args.grid_size ** 2
        self.network_context_manager = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_fp16 else NoOpContext()
        )

        # Compute mu for the scheduler (resolution-dependent)
        latent_h = args.latent_resolution
        latent_w = args.latent_resolution
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        sched_cfg = self.guidance_model.scheduler.config
        self._mu = calculate_shift(
            image_seq_len,
            sched_cfg.get("base_image_seq_len", 256),
            sched_cfg.get("max_image_seq_len", 4096),
            sched_cfg.get("base_shift", 0.5),
            sched_cfg.get("max_shift", 1.15),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Text encoding helpers
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, prompts: list, max_length: int = 512):
        """Encode a list of prompt strings → list of cap_feat tensors."""
        return encode_prompts(
            self.text_encoder, self.tokenizer, prompts,
            device=self.accelerator.device, max_length=max_length
        )

    @torch.no_grad()
    def get_uncond_cap_feats(self, batch_size: int, max_length: int = 512):
        """Get unconditional (empty prompt) cap_feats for CFG."""
        return self.encode_text([""] * batch_size, max_length=max_length)

    # ─────────────────────────────────────────────────────────────────────────
    # VAE helpers
    # ─────────────────────────────────────────────────────────────────────────

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel images."""
        vae = self.vae
        latents = latents.to(vae.dtype)
        # ZImage VAE uses shift_factor
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        images = vae.decode(latents, return_dict=False)[0].float()
        return images

    # ─────────────────────────────────────────────────────────────────────────
    # Generator forward
    # ─────────────────────────────────────────────────────────────────────────

    def run_generator(
        self,
        noise: torch.Tensor,
        cap_feats_list: list,
        compute_gradient: bool = True,
    ) -> torch.Tensor:
        """
        One-step generation: noise → x0_pred via the generator transformer.

        For training the generator, we condition at a high noise level (sigma ≈ 1).
        The input is treated as pure noise (or near-pure noise), and the generator
        predicts the clean image x0 directly.

        Args:
            noise: [B, C, H, W] Gaussian noise in latent space
            cap_feats_list: list of B text embedding tensors
            compute_gradient: if False, run under torch.no_grad()

        Returns:
            x0_pred: [B, C, H, W]
        """
        B = noise.shape[0]
        sigma = torch.full((B,), self.conditioning_sigma, device=noise.device)
        t_norms = torch.full((B,), self.conditioning_t_norm, device=noise.device)
        # Cast inputs to match generator's parameter dtype
        mdtype = next(self.feedforward_model.parameters()).dtype
        x_list = latents_to_list(noise.to(mdtype))
        cap_feats_list = [f.to(mdtype) for f in cap_feats_list]
        t_norms = t_norms.to(mdtype)

        ctx = self.network_context_manager if compute_gradient else torch.no_grad()
        if not compute_gradient and self.gradient_checkpointing:
            self.accelerator.unwrap_model(
                self.feedforward_model
            ).disable_gradient_checkpointing()

        with ctx:
            raw_list = self.feedforward_model(
                x_list, t_norms, cap_feats_list, return_dict=False
            )[0]

        if not compute_gradient and self.gradient_checkpointing:
            self.accelerator.unwrap_model(
                self.feedforward_model
            ).enable_gradient_checkpointing()

        raw_out = list_to_latents(raw_list)  # [B, C, H, W]

        # x0 = noise + sigma * raw_out  (flow matching: x0 = x_t + sigma * (x0 - noise))
        sigmas_bc = sigma.view(-1, 1, 1, 1).to(raw_out.dtype)
        if compute_gradient:
            x0_pred = (noise + sigmas_bc * raw_out).float()
        else:
            x0_pred = (noise + sigmas_bc * raw_out).float()
        return x0_pred

    # ─────────────────────────────────────────────────────────────────────────
    # Main forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        noise: torch.Tensor,
        cap_feats_list: list,
        uncond_cap_feats_list: list,
        visual: bool = False,
        compute_generator_gradient: bool = True,
        generator_turn: bool = False,
        guidance_turn: bool = False,
        guidance_data_dict: dict = None,
    ):
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn)

        if generator_turn:
            # ── Generator step ───────────────────────────────────────────────
            x0_pred = self.run_generator(
                noise, cap_feats_list,
                compute_gradient=compute_generator_gradient
            )

            generator_data_dict = {
                "latents": x0_pred,
                "cap_feats_list": cap_feats_list,
                "uncond_cap_feats_list": uncond_cap_feats_list,
            }

            if compute_generator_gradient:
                # Run guidance_model under no_grad so FSDP never registers
                # post-backward hooks for guidance params during the generator
                # backward pass.  Inside compute_distribution_matching_loss all
                # transformer calls are already under no_grad, so guidance params
                # were never in the graph anyway.  Without this wrapper, FSDP
                # (use_orig_params=True) registers hooks expecting gradients that
                # never arrive → _post_backward_hook_state AssertionError.
                with torch.no_grad():
                    _, log_dict = self.guidance_model(
                        generator_turn=True,
                        guidance_turn=False,
                        generator_data_dict=generator_data_dict,
                    )
                # Recompute loss_dm outside no_grad so gradient flows through
                # x0_pred back to the generator.  log_dict["dmtrain_grad"] is
                # already detached (a pure constant), so this is mathematically
                # identical to what compute_distribution_matching_loss returns.
                loss_dict = {}
                if not self.args.gan_alone:
                    grad = log_dict["dmtrain_grad"]
                    loss_dict["loss_dm"] = 0.5 * torch.nn.functional.mse_loss(
                        x0_pred.float(),
                        (x0_pred - grad).detach().float(),
                        reduction="mean",
                    )
            else:
                loss_dict = {}
                log_dict = {}

            if visual:
                with torch.no_grad():
                    if compute_generator_gradient and not self.args.gan_alone:
                        for key in ["dmtrain_pred_real_image", "dmtrain_pred_fake_image"]:
                            if key in log_dict:
                                latents = log_dict[key].detach()[:self.num_visuals]
                                log_dict[key + "_decoded"] = self.decode_latents(latents)

                    log_dict["generated_image"] = self.decode_latents(
                        x0_pred.detach()[:self.num_visuals]
                    )

            # Pass through guidance data dict for the guidance update step
            log_dict["guidance_data_dict"] = {
                "latents": x0_pred.detach(),
                "cap_feats_list": [f.detach() for f in cap_feats_list],
                "uncond_cap_feats_list": [f.detach() for f in uncond_cap_feats_list],
            }

        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict,
            )

        return loss_dict, log_dict
