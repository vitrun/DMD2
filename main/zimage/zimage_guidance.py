"""
ZImage DMD2 Guidance Model

Key differences from SD guidance:
1. Flow matching scheduler (FlowMatchEulerDiscreteScheduler) instead of DDIM
2. ZImageTransformer2DModel instead of UNet2DConditionModel
3. Transformer forward API: (x_list, t_norm, cap_feats_list)
   - x_list: list of [C, 1, H, W] tensors
   - t_norm: [B] float, (1000 - T) / 1000, where 0=noisy, 1=clean
   - cap_feats_list: list of [seq_len_i, cap_feat_dim] tensors (variable length)
4. x0 recovery from raw output: x0 = x_t + sigma * raw_transformer_output
   (raw output ≡ x0 - noise, the negative of the velocity used by scheduler)
5. Flow matching noise addition: x_t = (1 - sigma) * x0 + sigma * noise
"""
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from main.utils import DummyNetwork, NoOpContext
import torch.nn.functional as F
import torch.nn as nn
import torch


def latents_to_list(latents: torch.Tensor) -> list:
    """Convert [B, C, H, W] latents to list of [C, 1, H, W] tensors for ZImage transformer."""
    return [x.unsqueeze(1) for x in latents.unbind(0)]


def list_to_latents(out_list: list) -> torch.Tensor:
    """Convert list of [C, 1, H, W] tensors to batched [B, C, H, W] tensor."""
    return torch.stack([o.squeeze(1) for o in out_list], dim=0)


def predict_x0_flow(
    transformer,
    noisy_latents: torch.Tensor,
    cap_feats_list: list,
    t_norms: torch.Tensor,
    sigmas: torch.Tensor,
    guidance_scale: float = 1.0,
    uncond_cap_feats_list: list = None,
) -> torch.Tensor:
    """
    Run ZImage transformer and recover x0 via flow matching.

    The ZImageTransformer2DModel predicts (x0 - noise), i.e. the negative velocity.
    Recovery formula: x0_pred = x_t + sigma * raw_transformer_output

    Args:
        transformer: ZImageTransformer2DModel instance
        noisy_latents: [B, C, H, W] noisy latent tensors
        cap_feats_list: list of B text feature tensors, each [seq_len_i, cap_feat_dim]
        t_norms: [B] float tensor, normalized timestep = (1000 - T) / 1000
        sigmas: [B] float tensor, noise level used to add noise
        guidance_scale: CFG scale; if > 1, uncond_cap_feats_list must be provided
        uncond_cap_feats_list: list of B unconditional text feature tensors

    Returns:
        x0_pred: [B, C, H, W]
    """
    B = noisy_latents.shape[0]
    # Cast inputs to match the transformer's parameter dtype to avoid dtype mismatch.
    mdtype = next(transformer.parameters()).dtype
    x_list = latents_to_list(noisy_latents.to(mdtype))
    cap = [f.to(mdtype) for f in cap_feats_list]
    t = t_norms.to(mdtype)

    if guidance_scale > 1.0:
        assert uncond_cap_feats_list is not None, \
            "uncond_cap_feats_list required for guidance_scale > 1"
        uncond_cap = [f.to(mdtype) for f in uncond_cap_feats_list]
        # Double batch: [cond..., uncond...]
        combined_x = x_list + x_list
        combined_cap = cap + uncond_cap
        combined_t = torch.cat([t, t], dim=0)  # [2B]

        out_list = transformer(combined_x, combined_t, combined_cap, return_dict=False)[0]

        cond_out = list_to_latents(out_list[:B])    # [B, C, H, W]
        uncond_out = list_to_latents(out_list[B:])  # [B, C, H, W]
        raw_out = cond_out + guidance_scale * (cond_out - uncond_out)
    else:
        out_list = transformer(x_list, t, cap, return_dict=False)[0]
        raw_out = list_to_latents(out_list)  # [B, C, H, W]

    # x0 = x_t + sigma * raw_out  (derived from flow matching forward process)
    sigmas_bc = sigmas.view(-1, 1, 1, 1).to(noisy_latents.dtype)
    x0_pred = noisy_latents + sigmas_bc * raw_out.to(noisy_latents.dtype)
    return x0_pred


class ZImageGuidance(nn.Module):
    """
    Guidance model for ZImage DMD2 training.

    Contains:
    - real_transformer: frozen pretrained ZImage transformer (models real distribution)
    - fake_transformer: trainable copy (models generator's distribution)
    """

    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.use_fp16 = args.use_fp16
        self.real_guidance_scale = args.real_guidance_scale
        self.fake_guidance_scale = args.fake_guidance_scale
        assert self.fake_guidance_scale == 1, "No CFG for fake transformer training"

        self.gan_alone = args.gan_alone
        self.gradient_checkpointing = args.gradient_checkpointing

        # Real transformer: frozen reference for true distribution.
        # Load directly in target dtype to avoid float32 peak during conversion.
        real_dtype = torch.bfloat16 if args.use_fp16 else torch.float32
        self.real_transformer = ZImageTransformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=real_dtype
        )
        self.real_transformer.to(real_dtype)  # homogenize for FSDP (RMSNorm stays fp32 otherwise)
        self.real_transformer.requires_grad_(False)

        if self.gan_alone:
            del self.real_transformer

        # Fake transformer: trainable, learns the generator's distribution.
        # ZImage transformers are ~6B params; loading in bfloat16 saves ~24 GB vs float32
        # while maintaining sufficient gradient precision in practice.
        fake_dtype = torch.bfloat16 if args.use_fp16 else torch.float32
        self.fake_transformer = ZImageTransformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=fake_dtype
        )
        self.fake_transformer.to(fake_dtype)  # homogenize for FSDP (RMSNorm stays fp32 otherwise)
        self.fake_transformer.requires_grad_(True)

        # FSDP requires at least one module with dense (non-lazy) parameters.
        # Must match the dtype of the other params in the root FSDP unit (the
        # outer transformer embedder layers) to avoid the mixed-dtype flatten error.
        self.dummy_network = DummyNetwork()
        self.dummy_network.to(fake_dtype)
        self.dummy_network.requires_grad_(False)

        # Flow matching scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_id, subfolder="scheduler"
        )

        # Pre-compute training sigmas and normalized timesteps for the given resolution.
        # mu shifts the noise schedule based on image sequence length (ZImage dynamic shifting).
        latent_h = args.latent_resolution
        latent_w = args.latent_resolution
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        # Set up a dense set of training timesteps to sample from
        self.scheduler.set_timesteps(args.num_train_timesteps, device=accelerator.device, mu=mu)
        # sigmas[:-1] because the last sigma is 0.0 (fully clean, no noise)
        all_sigmas = self.scheduler.sigmas[:-1].clone()
        all_t_norms = ((1000 - self.scheduler.timesteps.float()) / 1000).clone()

        self.register_buffer("sigmas", all_sigmas)
        self.register_buffer("t_norms", all_t_norms)

        self.num_train_timesteps = len(all_sigmas)
        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)

        self.network_context_manager = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_fp16 else NoOpContext()
        )
        self.accelerator = accelerator

    def _sample_timesteps(self, batch_size: int, device):
        """Sample random step indices and return (sigmas, t_norms) for each sample."""
        step_indices = torch.randint(
            self.min_step,
            min(self.max_step + 1, self.num_train_timesteps),
            [batch_size],
            device=device,
            dtype=torch.long,
        )
        sigmas = self.sigmas[step_indices]     # [B]
        t_norms = self.t_norms[step_indices]   # [B]
        return sigmas, t_norms

    def _add_flow_noise(self, latents: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor):
        """Flow matching forward process: x_t = (1 - sigma) * x0 + sigma * noise."""
        sigmas_bc = sigmas.view(-1, 1, 1, 1)
        return (1 - sigmas_bc) * latents + sigmas_bc * noise

    def compute_distribution_matching_loss(
        self,
        latents: torch.Tensor,
        cap_feats_list: list,
        uncond_cap_feats_list: list,
    ):
        """
        DMD2 distribution matching loss.

        Gradient direction: (p_real - p_fake) / |p_real|
        where p_real = latent - x0_real_pred, p_fake = latent - x0_fake_pred
        """
        B = latents.shape[0]

        with torch.no_grad():
            sigmas, t_norms = self._sample_timesteps(B, latents.device)
            noise = torch.randn_like(latents)
            noisy_latents = self._add_flow_noise(latents, noise, sigmas)

            # Fake prediction: no CFG, full precision
            x0_fake = predict_x0_flow(
                self.fake_transformer,
                noisy_latents,
                cap_feats_list,
                t_norms,
                sigmas,
                guidance_scale=self.fake_guidance_scale,
            ).float()

            # Real prediction: with CFG, optionally bfloat16
            if self.use_fp16:
                real_dtype = torch.bfloat16
                real_noisy = noisy_latents.to(real_dtype)
                real_cap = [f.to(real_dtype) for f in cap_feats_list]
                real_uncond_cap = [f.to(real_dtype) for f in uncond_cap_feats_list]
            else:
                real_dtype = torch.float32
                real_noisy = noisy_latents
                real_cap = cap_feats_list
                real_uncond_cap = uncond_cap_feats_list

            x0_real = predict_x0_flow(
                self.real_transformer,
                real_noisy,
                real_cap,
                t_norms.to(real_dtype),
                sigmas.to(real_dtype),
                guidance_scale=self.real_guidance_scale,
                uncond_cap_feats_list=real_uncond_cap,
            ).float()

            p_real = latents.float() - x0_real
            p_fake = latents.float() - x0_fake

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = torch.nan_to_num(grad)

        # Pseudo-loss for backprop through generator
        loss = 0.5 * F.mse_loss(
            latents.float(),
            (latents - grad).detach().float(),
            reduction="mean"
        )

        loss_dict = {"loss_dm": loss}
        log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach().float(),
            "dmtrain_pred_real_image": x0_real.detach().float(),
            "dmtrain_pred_fake_image": x0_fake.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item(),
        }
        return loss_dict, log_dict

    def compute_loss_fake(
        self,
        latents: torch.Tensor,
        cap_feats_list: list,
    ):
        """
        Train the fake transformer to model the generator's distribution.

        Target: fake_transformer should recover x0 from noisy generator outputs.
        Loss: MSE between predicted raw output and the velocity target (x0 - noise).
        """
        if self.gradient_checkpointing:
            self.fake_transformer.enable_gradient_checkpointing()

        latents = latents.detach()
        B = latents.shape[0]

        sigmas, t_norms = self._sample_timesteps(B, latents.device)
        noise = torch.randn_like(latents)
        noisy_latents = self._add_flow_noise(latents, noise, sigmas)

        x_list = latents_to_list(noisy_latents)

        with self.network_context_manager:
            raw_pred_list = self.fake_transformer(
                x_list, t_norms, cap_feats_list, return_dict=False
            )[0]

        raw_pred = list_to_latents(raw_pred_list).float()  # [B, C, H, W]

        # Velocity target for flow matching: raw_output_target = x0 - noise
        # (equivalently: the transformer should predict x0 given x_t, i.e., raw = x0 - noise)
        raw_target = (latents - noise).float()

        loss_fake = torch.mean((raw_pred - raw_target) ** 2)

        loss_dict = {"loss_fake_mean": loss_fake}
        log_dict = {
            "faketrain_latents": latents.detach().float(),
            "faketrain_noisy_latents": noisy_latents.detach().float(),
        }

        if self.gradient_checkpointing:
            self.fake_transformer.disable_gradient_checkpointing()

        return loss_dict, log_dict

    def generator_forward(
        self,
        latents: torch.Tensor,
        cap_feats_list: list,
        uncond_cap_feats_list: list,
    ):
        loss_dict = {}
        log_dict = {}

        if not self.gan_alone:
            dm_dict, dm_log_dict = self.compute_distribution_matching_loss(
                latents, cap_feats_list, uncond_cap_feats_list
            )
            loss_dict.update(dm_dict)
            log_dict.update(dm_log_dict)

        return loss_dict, log_dict

    def guidance_forward(
        self,
        latents: torch.Tensor,
        cap_feats_list: list,
    ):
        fake_dict, fake_log_dict = self.compute_loss_fake(latents, cap_feats_list)
        return fake_dict, fake_log_dict

    def forward(
        self,
        generator_turn: bool = False,
        guidance_turn: bool = False,
        generator_data_dict: dict = None,
        guidance_data_dict: dict = None,
    ):
        if generator_turn:
            return self.generator_forward(
                latents=generator_data_dict["latents"],
                cap_feats_list=generator_data_dict["cap_feats_list"],
                uncond_cap_feats_list=generator_data_dict["uncond_cap_feats_list"],
            )
        elif guidance_turn:
            return self.guidance_forward(
                latents=guidance_data_dict["latents"],
                cap_feats_list=guidance_data_dict["cap_feats_list"],
            )
        else:
            raise NotImplementedError
