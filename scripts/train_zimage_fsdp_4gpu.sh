#!/bin/bash
# ZImage DMD2 FSDP training on 4 GPUs.
#
# Layer-allocation notes:
#   - fsdp_4gpu.yaml uses TRANSFORMER_BASED_WRAP (ZImageTransformerBlock),
#     so every transformer block is its own FSDP unit.  This avoids the
#     large un-wrapped outer-module shell that SIZE_BASED_WRAP leaves on
#     rank 0 and is the primary cause of device-0 OOM.
#   - fsdp_sync_module_states=false: each rank loads from CPU independently
#     so rank 0 never holds the full model on GPU before sharding.
#   - fsdp_use_orig_params=true: keeps unflattened param names, required for
#     gradient-checkpointing + FSDP to interoperate correctly.
#   - --gradient_checkpointing: trades compute for activation memory
#     (important given that text_encoder and VAE are unsharded on each GPU).
#
# Usage:
#   bash scripts/train_zimage_fsdp_4gpu.sh
# or override any arg:
#   OUTPUT_PATH=/my/path bash scripts/train_zimage_fsdp_4gpu.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Paths (override via env vars) ───────────────────────────────────────────
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/zimage_fsdp_4gpu/output}"
LOG_PATH="${LOG_PATH:-/tmp/zimage_fsdp_4gpu/log}"
CACHE_DIR="${CACHE_DIR:-/tmp/zimage_fsdp_4gpu/cache}"
TRAIN_PROMPT="${TRAIN_PROMPT:-$ROOT/test_prompts.txt}"
MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image}"

# ── W&B ─────────────────────────────────────────────────────────────────────
WANDB_ENTITY="${WANDB_ENTITY:-dummy}"
WANDB_PROJECT="${WANDB_PROJECT:-zimage_dmd2}"
WANDB_NAME="${WANDB_NAME:-fsdp_4gpu_$(date +%s)}"

mkdir -p "$OUTPUT_PATH" "$LOG_PATH" "$CACHE_DIR"

# Use exactly 4 GPUs.  Change the indices if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

WANDB_MODE=offline \
accelerate launch \
    --config_file "$ROOT/fsdp_configs/fsdp_4gpu.yaml" \
    "$ROOT/main/zimage/train_zimage.py" \
    --model_id                "$MODEL_ID" \
    --output_path             "$OUTPUT_PATH" \
    --log_path                "$LOG_PATH" \
    --cache_dir               "$CACHE_DIR" \
    --train_prompt_path       "$TRAIN_PROMPT" \
    --batch_size              1 \
    --train_iters             1000000 \
    --latent_resolution       128 \
    --latent_channel          16 \
    --resolution              1024 \
    --num_train_timesteps     50 \
    --min_step_percent        0.02 \
    --max_step_percent        0.98 \
    --real_guidance_scale     5.0 \
    --generator_lr            1e-5 \
    --guidance_lr             1e-5 \
    --use_fp16 \
    --fsdp \
    --gradient_checkpointing \
    --log_loss \
    --wandb_entity            "$WANDB_ENTITY" \
    --wandb_project           "$WANDB_PROJECT" \
    --wandb_name              "$WANDB_NAME" \
    --wandb_iters             100 \
    --log_iters               500 \
    --grid_size               2 \
    --max_checkpoint          10
