# src/neural_astar/utils/training.py
from __future__ import annotations

import random
import re
from glob import glob

import numpy as np
import logging

import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from neural_astar.planner.astar import VanillaAstar
from neural_astar.utils.geometric_losses import (
    GoalCentricGeodesicLoss,
    VectorFieldLoss,
)


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """Load model weights from PyTorch Lightning checkpoint."""
    ckpt_file = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch.load(ckpt_file, weights_only=True)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted


class PlannerModule(pl.LightningModule):
    """PyTorch Lightning module for Neural A* training."""
    
    def __init__(
        self,
        planner,
        config,
        use_guidance: bool = False,
        geo_supervision: bool = False,
    ):
        super().__init__()
        self.planner = planner
        self.vanilla_astar = VanillaAstar()
        self.config = config
        self.geo_supervision = geo_supervision
        self.geo_loss_weight = getattr(config.params, "geo_loss_weight", 0.1)
        self.geo_warmup_epochs = getattr(config.params, "geo_warmup_epochs", 5)
        self.vec_loss_weight = getattr(config.params, "vec_loss_weight", 0.1)
        temps = getattr(config.params, "geo_temps", [0.5, 1.0, 2.0])
        self.geo_loss_fn = GoalCentricGeodesicLoss(temperatures=list(temps))
        self.vec_loss_fn = VectorFieldLoss(loss_type="cosine")

    def forward(self, map_designs, start_maps, goal_maps):
        return self.planner(map_designs, start_maps, goal_maps)

    def configure_optimizers(self):
        if not self.geo_supervision:
            return torch.optim.RMSprop(
                self.planner.parameters(),
                self.config.params.lr,
            )

        optimizer = torch.optim.AdamW(
            self.planner.parameters(), 
            lr=self.config.params.lr,
            weight_decay=0.01
        )
        
        try:
            # OneCycleLR with warmup (preferred)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.params.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # 10% warmup
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
        except (RuntimeError, MisconfigurationException) as e:
            # Fallback for DDP, variable accumulate_grad_batches, or DeepSpeed
            logging.warning(
                f"estimated_stepping_batches unavailable ({e}). "
                "Falling back to CosineAnnealingLR."
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.params.num_epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }

    def training_step(self, train_batch, batch_idx):
        if self.geo_supervision:
            if len(train_batch) == 8:
                (
                    map_designs,
                    start_maps,
                    goal_maps,
                    opt_trajs,
                    dist,
                    reachable,
                    vx,
                    vy,
                ) = train_batch
            else:
                map_designs, start_maps, goal_maps, opt_trajs, dist, reachable = (
                    train_batch
                )
                vx = vy = None
            outputs = self.forward(map_designs, start_maps, goal_maps)
            path_loss = F.l1_loss(outputs.histories, opt_trajs)

            attn_weights = self.planner.get_attention_weights()
            geo_loss = 0.0
            if attn_weights:
                geo_loss = self.geo_loss_fn(attn_weights[-1], dist, reachable)
            warmup_factor = 1.0
            if self.geo_warmup_epochs > 0:
                warmup_factor = min(
                    1.0, float(self.current_epoch) / float(self.geo_warmup_epochs)
                )
            effective_geo_weight = self.geo_loss_weight * warmup_factor
            vec_loss = 0.0
            effective_vec_weight = self.vec_loss_weight * warmup_factor
            vector_field = (
                self.planner.get_vector_field()
                if hasattr(self.planner, "get_vector_field")
                else None
            )
            if (
                vx is not None
                and vy is not None
                and vector_field is not None
                and vector_field[0] is not None
            ):
                pred_vx, pred_vy = vector_field
                vec_loss = self.vec_loss_fn(pred_vx, pred_vy, vx, vy, reachable)
            total_loss = (
                path_loss
                + effective_geo_weight * geo_loss
                + effective_vec_weight * vec_loss
            )
            self.log(
                "metrics/train_loss",
                total_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("metrics/train_path_loss", path_loss)
            if isinstance(geo_loss, torch.Tensor):
                self.log("metrics/train_geo_loss", geo_loss)
            if isinstance(vec_loss, torch.Tensor):
                self.log("metrics/train_vec_loss", vec_loss)
            return total_loss

        map_designs, start_maps, goal_maps, opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = F.l1_loss(outputs.histories, opt_trajs)
        self.log(
            "metrics/train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.geo_supervision:
            if len(val_batch) == 8:
                (
                    map_designs,
                    start_maps,
                    goal_maps,
                    opt_trajs,
                    dist,
                    reachable,
                    vx,
                    vy,
                ) = val_batch
            else:
                map_designs, start_maps, goal_maps, opt_trajs, dist, reachable = (
                    val_batch
                )
                vx = vy = None
            outputs = self.forward(map_designs, start_maps, goal_maps)
            path_loss = F.l1_loss(outputs.histories, opt_trajs)
            attn_weights = self.planner.get_attention_weights()
            geo_loss = 0.0
            if attn_weights:
                geo_loss = self.geo_loss_fn(attn_weights[-1], dist, reachable)
            vec_loss = 0.0
            warmup_factor = 1.0
            if self.geo_warmup_epochs > 0:
                warmup_factor = min(
                    1.0, float(self.current_epoch) / float(self.geo_warmup_epochs)
                )
            effective_vec_weight = self.vec_loss_weight * warmup_factor
            vector_field = (
                self.planner.get_vector_field()
                if hasattr(self.planner, "get_vector_field")
                else None
            )
            if (
                vx is not None
                and vy is not None
                and vector_field is not None
                and vector_field[0] is not None
            ):
                pred_vx, pred_vy = vector_field
                vec_loss = self.vec_loss_fn(pred_vx, pred_vy, vx, vy, reachable)
            loss = (
                path_loss
                + self.geo_loss_weight * geo_loss
                + effective_vec_weight * vec_loss
            )
            self.log(
                "metrics/val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("metrics/val_path_loss", path_loss)
            if isinstance(geo_loss, torch.Tensor):
                self.log("metrics/val_geo_loss", geo_loss)
            if isinstance(vec_loss, torch.Tensor):
                self.log("metrics/val_vec_loss", vec_loss)
            vanilla_map = map_designs
        else:
            map_designs, start_maps, goal_maps, opt_trajs = val_batch
            outputs = self.forward(map_designs, start_maps, goal_maps)
            loss = F.l1_loss(outputs.histories, opt_trajs)
            self.log(
                "metrics/val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
            )
            vanilla_map = map_designs

        if vanilla_map.shape[1] == 1:
            va_outputs = self.vanilla_astar(vanilla_map, start_maps, goal_maps)
            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()

            exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

            self.log("metrics/p_opt", p_opt)
            self.log("metrics/p_exp", p_exp)
            self.log("metrics/h_mean", h_mean)

        return loss


def set_global_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
