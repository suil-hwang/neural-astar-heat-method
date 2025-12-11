# scripts/train.py
"""Training Neural A* 
Author: Ryo Yonetani
Affiliation: OSX

Gated Fusion 사용 방법:
- mode=ours + encoder.arch=GatedUnet
- 7채널 입력을 인코더 내부에서 Base(3ch)/Heat(4ch)로 분리 후 Gate 융합
"""
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
    else:
        encoder_input = config.encoder.input  # Default: "m+"
        geo_supervision = False
    
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

    # 인코더 아키텍처: config에서 지정 (GeoAttentionUnet/GeoAttentionCNN/Unet/CNN)
    encoder_arch = config.encoder.arch
    if mode == "ours" and encoder_arch == "CNN":
        # Default fallback: ours+CNN 은 GeoAttentionUnet 으로 대체
        encoder_arch = "GeoAttentionUnet"
    const_val = getattr(config, 'const', None)
    
    # 모델 생성 (단일 인터페이스)
    neural_astar = NeuralAstar(
        encoder_input=encoder_input,
        encoder_arch=encoder_arch,
        encoder_depth=config.encoder.depth,
        learn_obstacles=False,
        const=const_val,
        Tmax=config.Tmax,
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=True, mode="max"
    )
    
    # 학습 모드 정보 출력
    print("=" * 60)
    print(f"Training mode: {mode}")
    print(f"Encoder: {encoder_arch} (input channels: {len(encoder_input)})")
    
    if mode == "ours":
        print("[Geo-Attention] 3ch 입력 + geodesic supervision (goal-centric)")
        print("  - Heat Method 결과는 teacher로만 사용 (추론 시 불필요)")
        print("  - Attention을 geodesic 분포(KL)로 감독")
    elif "Gated" in encoder_arch:
        print("[Gated Fusion] 입력을 내부에서 분리 처리")
        print("  - Base: Map(1) + Start(1) + Goal(1)  → 국소적 정보")
        print("  - Heat: VecX + VecY + Dist + Reachable  → 전역적 흐름")
        print("  - Gate로 상황에 맞게 동적 융합")
        print("  - planner.get_gate_map()으로 시각화 가능")
    else:
        print(f"[Standard] 모든 채널 Concatenate 후 단일 인코더")
    
    if mode == "ours_geo_only":
        print("[Ablation] Vector field 제거 (5채널)")
    print("=" * 60)

    module = PlannerModule(
        neural_astar,
        config,
        use_guidance=False,
        geo_supervision=geo_supervision,
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

# 2. Geo-Attention 
# python scripts/train.py mode=ours encoder.arch=GeoAttentionCNN
# python scripts/train.py mode=ours encoder.arch=GeoAttentionUnet