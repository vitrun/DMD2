"""
ZImage DMD2 Training Script

Follows the same Trainer → UniModel architecture as train_sd.py, adapted for:
- ZImageTransformer2DModel (instead of UNet2DConditionModel)
- FlowMatchEulerDiscreteScheduler / flow matching (instead of DDIM)
- Chat-template text encoding → variable-length cap_feats_list per batch
- 16-channel VAE latents (instead of 4-channel)

Usage example:
  torchrun --nproc_per_node=8 main/zimage/train_zimage.py \\
    --model_id Z-a-o/Z-Image-Turbo \\
    --output_path /path/to/output \\
    --train_prompt_path /path/to/prompts.txt \\
    --latent_resolution 128 \\
    --latent_channel 16 \\
    --resolution 1024 \\
    --use_fp16 \\
    --wandb_name my_zimage_dmd2
"""
import matplotlib
matplotlib.use('Agg')

from main.utils import prepare_images_for_saving, draw_valued_array, cycle
from main.zimage.zimage_unified_model import ZImageUniModel
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import argparse
import pickle
import shutil
import wandb
import torch
import time
import os


class ZImageTextDataset(Dataset):
    """
    Dataset that returns raw text prompts.

    ZImage text encoding happens at training time (in encode_text) because the
    chat template produces variable-length sequences that cannot be easily batched
    as fixed-length tensors in advance.
    """

    def __init__(self, anno_path: str):
        if anno_path.endswith(".txt"):
            self.prompts = []
            with open(anno_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.prompts.append(line)
        else:
            self.prompts = pickle.load(open(anno_path, "rb"))

        print(f"[ZImageTextDataset] Loaded {len(self.prompts)} prompts from {anno_path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        if prompt is None:
            prompt = ""
        return {"prompt": prompt, "index": idx}


def collate_text(batch):
    """Collate a list of text-only samples into a dict with a list of prompts."""
    prompts = [item["prompt"] for item in batch]
    indices = [item["index"] for item in batch]
    return {"prompt": prompts, "index": indices}


class Trainer:
    def __init__(self, args):
        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        accelerator_project_config = ProjectConfiguration(logging_dir=args.log_path)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
            project_config=accelerator_project_config,
        )
        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        if accelerator.is_main_process:
            output_path = os.path.join(
                args.output_path, f"time_{int(time.time())}_seed{args.seed}"
            )
            os.makedirs(output_path, exist_ok=False)

            self.cache_dir = os.path.join(
                args.cache_dir, f"time_{int(time.time())}_seed{args.seed}"
            )
            os.makedirs(self.cache_dir, exist_ok=False)
            self.output_path = output_path

            os.makedirs(args.log_path, exist_ok=True)

            run = wandb.init(
                config=args,
                dir=args.log_path,
                entity=args.wandb_entity,
                project=args.wandb_project,
            )
            wandb.run.log_code(".")
            wandb.run.name = args.wandb_name
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

        # ── Build unified model ──────────────────────────────────────────────
        self.model = ZImageUniModel(args, accelerator)
        self.max_grad_norm = args.max_grad_norm
        self.step = 0

        # Load checkpoints if provided
        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"Loading ckpt_only from {args.ckpt_only_path}")
            gen_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guid_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")
            print(self.model.feedforward_model.load_state_dict(
                torch.load(gen_path, map_location="cpu"), strict=False
            ))
            print(self.model.guidance_model.load_state_dict(
                torch.load(guid_path, map_location="cpu"), strict=False
            ))
            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"Loading generator ckpt from {args.generator_ckpt_path}")
            print(self.model.feedforward_model.load_state_dict(
                torch.load(args.generator_ckpt_path, map_location="cpu"), strict=True
            ))

        # ── Datasets and dataloaders ─────────────────────────────────────────
        dataset = ZImageTextDataset(args.train_prompt_path)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_text,
        )
        dataloader = accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        guidance_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_text,
        )
        guidance_dataloader = accelerator.prepare(guidance_dataloader)
        self.guidance_dataloader = cycle(guidance_dataloader)

        self.fsdp = args.fsdp

        # ── FSDP parameter sync ──────────────────────────────────────────────
        if self.fsdp and (args.ckpt_only_path is None):
            gen_path = os.path.join(
                args.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model.bin"
            )
            guid_path = os.path.join(
                args.output_path, f"checkpoint_model_{self.step:06d}", "pytorch_model_1.bin"
            )
            if accelerator.is_main_process:
                print("Saving initial ckpt for FSDP parameter sync across nodes")
                os.makedirs(
                    os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}"),
                    exist_ok=True,
                )
                torch.save(self.model.feedforward_model.state_dict(), gen_path)
                torch.save(self.model.guidance_model.state_dict(), guid_path)

            accelerator.wait_for_everyone()
            print(self.model.feedforward_model.load_state_dict(
                torch.load(gen_path, map_location="cpu"), strict=True
            ))
            print(self.model.guidance_model.load_state_dict(
                torch.load(guid_path, map_location="cpu"), strict=True
            ))
            if accelerator.is_main_process:
                print("FSDP parameter sync done")

        # ── Accelerator prepare (models) ─────────────────────────────────────
        # Prepare models first so optimizer references DDP/FSDP-wrapped params.
        (
            self.model.feedforward_model,
            self.model.guidance_model,
        ) = accelerator.prepare(
            self.model.feedforward_model,
            self.model.guidance_model,
        )

        # ── Optimizers ───────────────────────────────────────────────────────
        self.optimizer_generator = torch.optim.AdamW(
            [p for p in self.model.feedforward_model.parameters() if p.requires_grad],
            lr=args.generator_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.optimizer_guidance = torch.optim.AdamW(
            [p for p in self.model.guidance_model.parameters() if p.requires_grad],
            lr=args.guidance_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )

        (
            self.optimizer_generator, self.optimizer_guidance,
            self.scheduler_generator, self.scheduler_guidance,
        ) = accelerator.prepare(
            self.optimizer_generator, self.optimizer_guidance,
            self.scheduler_generator, self.scheduler_guidance,
        )

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.latent_resolution = args.latent_resolution
        self.grid_size = args.grid_size
        self.log_loss = args.log_loss
        self.latent_channel = args.latent_channel
        self.no_save = args.no_save
        self.max_checkpoint = args.max_checkpoint
        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def fsdp_state_dict(self, model):
        policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
            return model.state_dict()

    def load(self, checkpoint_path):
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def save(self):
        if self.fsdp:
            gen_sd = self.fsdp_state_dict(self.model.feedforward_model)
            guid_sd = self.fsdp_state_dict(self.model.guidance_model)

        if self.accelerator.is_main_process:
            out_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(out_path, exist_ok=True)
            print(f"Saving checkpoint to {out_path}")

            if self.fsdp:
                torch.save(gen_sd, os.path.join(out_path, "pytorch_model.bin"))
                del gen_sd
                torch.save(guid_sd, os.path.join(out_path, "pytorch_model_1.bin"))
                del guid_sd
            else:
                self.accelerator.save_state(out_path)

            # Keep only the latest checkpoint in output_path
            for folder in os.listdir(self.output_path):
                if (
                    folder.startswith("checkpoint_model")
                    and folder != f"checkpoint_model_{self.step:06d}"
                ):
                    shutil.rmtree(os.path.join(self.output_path, folder))

            # Copy to cache (keeps up to max_checkpoint checkpoints)
            cache_target = os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")
            if os.path.exists(cache_target):
                shutil.rmtree(cache_target)
            shutil.copytree(out_path, cache_target)

            cached = sorted(
                f for f in os.listdir(self.cache_dir) if f.startswith("checkpoint_model")
            )
            for old in cached[: -self.max_checkpoint]:
                shutil.rmtree(os.path.join(self.cache_dir, old))

            print("Checkpoint saved.")
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Training step
    # ─────────────────────────────────────────────────────────────────────────

    def train_one_step(self):
        self.model.train()
        accelerator = self.accelerator

        # 16-channel latents (ZImage VAE)
        noise = torch.randn(
            self.batch_size,
            self.latent_channel,
            self.latent_resolution,
            self.latent_resolution,
            device=accelerator.device,
        )

        visual = self.step % self.wandb_iters == 0
        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0

        if COMPUTE_GENERATOR_GRADIENT:
            batch = next(self.dataloader)
        else:
            batch = next(self.guidance_dataloader)

        prompts = batch["prompt"]  # list of strings

        # Encode text (no gradient; text encoder is frozen)
        cap_feats_list = self.model.encode_text(prompts)
        uncond_cap_feats_list = self.model.get_uncond_cap_feats(len(prompts))

        # ── Generator step ───────────────────────────────────────────────────
        generator_loss_dict, generator_log_dict = self.model(
            noise,
            cap_feats_list,
            uncond_cap_feats_list,
            visual=visual,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=True,
            guidance_turn=False,
        )

        generator_loss = 0.0
        if COMPUTE_GENERATOR_GRADIENT:
            if not self.args.gan_alone:
                generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight

            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(
                self.model.feedforward_model.parameters(), self.max_grad_norm
            )
            self.optimizer_generator.step()
            self.optimizer_generator.zero_grad()
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()

        # ── Guidance step (fake transformer) ─────────────────────────────────
        guidance_loss_dict, guidance_log_dict = self.model(
            noise,
            cap_feats_list,
            uncond_cap_feats_list,
            visual=visual,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict["guidance_data_dict"],
        )

        guidance_loss = guidance_loss_dict["loss_fake_mean"]
        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(
            self.model.guidance_model.parameters(), self.max_grad_norm
        )
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.optimizer_generator.zero_grad()
        self.scheduler_guidance.step()

        # ── Logging ──────────────────────────────────────────────────────────
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        generated_latents = log_dict["guidance_data_dict"]["latents"]
        gen_mean = accelerator.gather(generated_latents.mean()).mean()
        gen_std = accelerator.gather(generated_latents.std()).mean()

        if COMPUTE_GENERATOR_GRADIENT and not self.args.gan_alone:
            real_pred = log_dict.get("dmtrain_pred_real_image")
            fake_pred = log_dict.get("dmtrain_pred_fake_image")
            if real_pred is not None:
                real_mean = accelerator.gather(real_pred.mean()).mean()
                fake_mean = accelerator.gather(fake_pred.mean()).mean()

        if accelerator.is_main_process and self.log_loss and not visual:
            wandb_dict = {
                "loss_fake_mean": guidance_loss_dict["loss_fake_mean"].item(),
                "guidance_grad_norm": guidance_grad_norm.item(),
                "generated_latent_mean": gen_mean.item(),
                "generated_latent_std": gen_std.item(),
                "batch_size": len(noise),
            }
            if COMPUTE_GENERATOR_GRADIENT:
                wandb_dict["generator_grad_norm"] = generator_grad_norm.item()
                if not self.args.gan_alone:
                    wandb_dict["loss_dm"] = loss_dict["loss_dm"].item()
                    wandb_dict["dmtrain_gradient_norm"] = log_dict.get("dmtrain_gradient_norm", 0)
            wandb.log(wandb_dict, step=self.step)

        if visual:
            if not self.args.gan_alone:
                for key in ["dmtrain_pred_real_image_decoded", "dmtrain_pred_fake_image_decoded"]:
                    if key in log_dict:
                        log_dict[key] = accelerator.gather(log_dict[key])

            if "generated_image" in log_dict:
                log_dict["generated_image"] = accelerator.gather(log_dict["generated_image"])

        if accelerator.is_main_process and visual:
            with torch.no_grad():
                data_dict = {}

                if not self.args.gan_alone:
                    for key, label in [
                        ("dmtrain_pred_real_image_decoded", "dmtrain_pred_real_image"),
                        ("dmtrain_pred_fake_image_decoded", "dmtrain_pred_fake_image"),
                    ]:
                        if key in log_dict:
                            grid = prepare_images_for_saving(
                                log_dict[key], resolution=self.resolution, grid_size=self.grid_size
                            )
                            data_dict[label] = wandb.Image(grid)

                    if "loss_dm" in loss_dict:
                        data_dict["loss_dm"] = loss_dict["loss_dm"].item()
                    if "dmtrain_gradient_norm" in log_dict:
                        data_dict["dmtrain_gradient_norm"] = log_dict["dmtrain_gradient_norm"]

                if "generated_image" in log_dict:
                    grid = prepare_images_for_saving(
                        log_dict["generated_image"],
                        resolution=self.resolution,
                        grid_size=self.grid_size,
                    )
                    data_dict["generated_image"] = wandb.Image(grid)

                data_dict.update({
                    "loss_fake_mean": loss_dict["loss_fake_mean"].item(),
                    "generator_grad_norm": generator_grad_norm.item() if COMPUTE_GENERATOR_GRADIENT else 0,
                    "guidance_grad_norm": guidance_grad_norm.item(),
                })

                wandb.log(data_dict, step=self.step)

        # ── Stdout progress ──────────────────────────────────────────────────
        if accelerator.is_main_process:
            parts = [f"step {self.step}"]
            if "loss_dm" in loss_dict:
                parts.append(f"loss_dm={loss_dict['loss_dm'].item():.4f}")
            parts.append(f"loss_fake={guidance_loss_dict['loss_fake_mean'].item():.4f}")
            if COMPUTE_GENERATOR_GRADIENT:
                parts.append(f"gen_gnorm={generator_grad_norm.item():.3f}")
            parts.append(f"guid_gnorm={guidance_grad_norm.item():.3f}")
            parts.append(f"lat_mean={gen_mean.item():.3f}")
            print("  ".join(parts), flush=True)

        self.accelerator.wait_for_everyone()

    def train(self):
        for _ in range(self.step, self.train_iters):
            self.train_one_step()

            if not self.no_save and self.step % self.log_iters == 0:
                self.save()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                current_time = time.time()
                if hasattr(self, "_prev_time"):
                    wandb.log({"per_iteration_time": current_time - self._prev_time}, step=self.step)
                self._prev_time = current_time

            self.step += 1


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ZImage DMD2 Training")

    # Model
    parser.add_argument("--model_id", type=str, default="Tongyi-MAI/Z-Image",
                        help="HuggingFace model ID or local path for ZImagePipeline")

    # Paths
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--train_prompt_path", type=str, required=True,
                        help=".txt file with one prompt per line, or .pkl list")
    parser.add_argument("--cache_dir", type=str, default="/tmp/zimage_dmd2_cache")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Resume from full Accelerate checkpoint")
    parser.add_argument("--ckpt_only_path", type=str, default=None,
                        help="Resume from model-weights-only checkpoint")
    parser.add_argument("--generator_ckpt_path", type=str, default=None,
                        help="Load generator weights only")

    # Training
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--warmup_step", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1,
                        help="Update generator every N guidance updates")

    # Learning rates
    parser.add_argument("--generator_lr", type=float, default=1e-5)
    parser.add_argument("--guidance_lr", type=float, default=1e-5)

    # Image / latent dimensions
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Pixel resolution of generated images")
    parser.add_argument("--latent_resolution", type=int, default=128,
                        help="Latent spatial resolution (pixel_res // 8 for standard VAE)")
    parser.add_argument("--latent_channel", type=int, default=16,
                        help="Number of VAE latent channels (16 for ZImage)")

    # Scheduler / noise
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of timesteps to pre-compute for training sigma sampling")
    parser.add_argument("--min_step_percent", type=float, default=0.02)
    parser.add_argument("--max_step_percent", type=float, default=0.98)
    parser.add_argument("--conditioning_sigma_index", type=int, default=0,
                        help="Index into pre-computed sigmas for 1-step conditioning (0=max noise)")

    # CFG
    parser.add_argument("--real_guidance_scale", type=float, default=5.0,
                        help="CFG scale for real transformer predictions")
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0,
                        help="Must be 1.0 (no CFG for fake transformer)")

    # Losses
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)
    parser.add_argument("--gan_alone", action="store_true",
                        help="Skip DM loss; use only fake-transformer training (for debugging)")

    # Precision and memory
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use bfloat16 for real transformer; generator stays float32")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Logging
    parser.add_argument("--log_iters", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--wandb_iters", type=int, default=100,
                        help="Log visuals to W&B every N steps")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--no_save", action="store_true",
                        help="Skip saving checkpoints (for debugging)")
    parser.add_argument("--log_loss", action="store_true",
                        help="Log loss scalars every step (not just on visual steps)")
    parser.add_argument("--grid_size", type=int, default=2,
                        help="Visual grid is grid_size x grid_size images")
    parser.add_argument("--max_checkpoint", type=int, default=30,
                        help="Max number of checkpoints to keep in cache")

    # Text encoding
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max token length for text encoding")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and hasattr(args, "local_rank"):
        args.local_rank = env_local_rank

    assert args.gradient_accumulation_steps == 1, "Gradient accumulation not yet supported"
    assert args.fake_guidance_scale == 1.0, "fake_guidance_scale must be 1.0"
    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, \
        "wandb_iters must be a multiple of dfake_gen_update_ratio"

    return args


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
