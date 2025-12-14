# scripts/train.py
from __future__ import annotations

import os
import sys
import warnings

import hydra
import pytorch_lightning as pl
import torch

# Prefer local src/ over any globally installed neural_astar
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint

# Suppress warnings
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message=".*num_workers.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
warnings.filterwarnings("ignore", message=".*Precision.*mixed.*")

# Set float32 matmul precision for Tensor Cores
torch.set_float32_matmul_precision('high')


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(config):

    # Set random seed
    set_global_seeds(config.seed)
    
    # Determine mode (only two modes supported)
    mode = getattr(config, "mode", "neural_astar")
    if mode not in ("neural_astar", "ours"):
        raise ValueError(
            f"Unsupported mode={mode!r}. Supported modes: neural_astar, ours."
        )

    # Determine supervision mode
    geo_supervision = mode == "ours"
    encoder_input = "msg" if geo_supervision else config.encoder.input  # Default: "m+"
    
    # Get num_workers from config
    num_workers = getattr(config.params, 'num_workers', 4)
    
    train_loader = create_dataloader(
        config.dataset + ".npz",
        "train",
        config.params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        geo_supervision=geo_supervision,
    )
    val_loader = create_dataloader(
        config.dataset + ".npz",
        "valid",
        config.params.batch_size,
        shuffle=False,
        num_workers=num_workers,
        geo_supervision=geo_supervision,
    )

    encoder_arch = str(config.encoder.arch)
    if geo_supervision and encoder_arch != "MultiHeadGeoUnet":
        print(
            f"Warning: mode=ours but encoder.arch={encoder_arch!r}. "
            "Recommended: MultiHeadGeoUnet"
        )

    # Optional encoder kwargs (e.g., backbone for Unet variants)
    encoder_kwargs = {}
    backbone = getattr(config.encoder, "backbone", None)
    if backbone is not None and encoder_arch == "MultiHeadGeoUnet":
        encoder_kwargs["backbone"] = backbone
    const_val = getattr(config, 'const', None)
    
    encoder_depth = int(config.encoder.depth)
    if encoder_depth < 1:
        raise ValueError(f"encoder.depth must be >= 1, got {encoder_depth}")

    neural_astar = NeuralAstar(
        encoder_input=encoder_input,
        encoder_arch=encoder_arch,
        encoder_depth=encoder_depth,
        learn_obstacles=False,
        const=const_val,
        Tmax=config.Tmax,
        encoder_kwargs=encoder_kwargs,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=True, mode="max"
    )
    
    # Extract map size from dataset for Eikonal loss scaling
    map_size = train_loader.dataset.map_designs.shape[-1]
    
    print("=" * 60)
    print(f"Training mode: {mode}")
    print(f"Encoder: {encoder_arch} (input channels: {len(encoder_input)})")
    print(f"Encoder depth: {encoder_depth}")
    print(f"Map size: {map_size}x{map_size}")
    print("=" * 60)

    module = PlannerModule(
        neural_astar,
        config,
        use_guidance=False,
        direct_geo_supervision=geo_supervision,
        map_size=map_size,
    )
    logdir = f"{config.logdir}/{mode}/{os.path.basename(config.dataset)}"
    enable_progress_bar = sys.stdout.isatty()
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",  # BF16 is optimized for RTX 30/40 series and H100
        benchmark=True,  # Enable cuDNN autotuner for consistent input sizes
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=True,  # Use default model summary
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()

# ============================================================================
# Usage Examples
# ============================================================================

# 1. Neural A* 
# python scripts/train.py mode=neural_astar encoder.arch=Unet 

# Tensorboard
# tensorboard --logdir model/neural_astar/mazes_032_moore_c8_ours
# tensorboard --logdir model/neural_astar/all_064_moore_c16_ours
# tensorboard --logdir model/neural_astar/mixed_064_moore_c16_ours

# 2. Ours 
# python scripts/train.py mode=ours encoder.arch=MultiHeadGeoUnet

# Tensorboard
# tensorboard --logdir model/ours/mazes_032_moore_c8_ours
# tensorboard --logdir model/ours/all_064_moore_c16_ours
# tensorboard --logdir model/ours/mixed_064_moore_c16_ours
