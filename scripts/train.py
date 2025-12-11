# scripts/train.py
from __future__ import annotations

import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint

# Suppress warnings
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message=".*num_workers.*")

# Set float32 matmul precision for Tensor Cores
torch.set_float32_matmul_precision('high')


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(config):

    # Set random seed
    set_global_seeds(config.seed)
    
    # Determine mode
    mode = getattr(config, 'mode', 'neural_astar')
    
    # For vanilla mode, skip training
    if mode == "vanilla":
        print("=" * 60)
        print("Vanilla A* Mode: No training needed.")
        print("Use vanilla_astar for evaluation only.")
        print("=" * 60)
        return
    
    # Set encoder input and supervision flags based on mode
    if mode == "ours":
        encoder_input = "msg"
        geo_supervision = True
        use_phi = True
    else:
        encoder_input = "m+"
        geo_supervision = False
        use_phi = False
    
    # Get num_workers from config
    num_workers = getattr(config.params, 'num_workers', 4)
    
    num_anchors = getattr(config.params, "num_anchors", getattr(config.encoder, "num_anchors", 8))

    train_loader = create_dataloader(
        config.dataset + ".npz",
        "train",
        config.params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        geo_supervision=geo_supervision,
        use_phi=use_phi,
        num_anchors=num_anchors,
    )
    val_loader = create_dataloader(
        config.dataset + ".npz",
        "valid",
        config.params.batch_size,
        shuffle=False,
        num_workers=num_workers,
        geo_supervision=geo_supervision,
        use_phi=use_phi,
        num_anchors=num_anchors,
    )

    encoder_arch = config.encoder.arch
    if mode == "ours" and not encoder_arch.startswith("GeoAttention"):
        raise ValueError(
            f"mode=ours requires GeoAttention*, got '{encoder_arch}'"
        )
    const_val = getattr(config, 'const', None)
    
    encoder_kwargs = {}
    encoder_kwargs["num_anchors"] = num_anchors
    encoder_kwargs["attention_heads"] = getattr(config.encoder, "attention_heads", 4)
    encoder_kwargs["num_attention_blocks"] = getattr(
        config.encoder, "num_attention_blocks", 2
    )
    encoder_kwargs["init_beta"] = getattr(config.encoder, "init_beta", 0.5)
    encoder_kwargs["use_geodesic_pe"] = getattr(config.encoder, "use_geodesic_pe", True)
    encoder_kwargs["predict_vector_field"] = getattr(
        config.encoder, "predict_vector_field", True
    )
    encoder_kwargs["use_local_attention"] = getattr(
        config.encoder, "use_local_attention", False
    )
    encoder_kwargs["local_window_size"] = getattr(
        config.encoder, "local_window_size", 7
    )

    neural_astar = NeuralAstar(
        encoder_input=encoder_input,
        encoder_arch=encoder_arch,
        encoder_depth=config.encoder.depth,
        learn_obstacles=False,
        const=const_val,
        Tmax=config.Tmax,
        encoder_kwargs=encoder_kwargs if mode == "ours" else None,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=True, mode="max"
    )
    
    print("=" * 60)
    print(f"Training mode: {mode}")
    print(f"Encoder: {encoder_arch} (input channels: {len(encoder_input)})")
    
    if mode == "ours":
        print("[Geo-Attention Neural A*]")
    else:
        print("[Standard Neural A*]")
    print("=" * 60)

    module = PlannerModule(
        neural_astar,
        config,
        use_guidance=False,
        geo_supervision=geo_supervision,
        use_phi=use_phi,
        num_anchors=num_anchors,
    )
    logdir = f"{config.logdir}/{mode}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()

# ============================================================================
# Usage Examples
# ============================================================================

# 1. Standard Neural A* 
# python scripts/train.py mode=neural_astar encoder.arch=CNN
# python scripts/train.py mode=neural_astar encoder.arch=Unet

# 2. Geo-Attention Neural A*
# python scripts/train.py mode=ours encoder.arch=GeoAttentionCNN
# python scripts/train.py mode=ours encoder.arch=GeoAttentionUnet