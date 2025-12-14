# src/neural_astar/utils/training.py
from __future__ import annotations
import sys

import random
import re
from glob import glob

import numpy as np
import logging

import pytorch_lightning as pl

# Disable torch.compile/dynamo on Windows (Triton is not supported)
if sys.platform == "win32":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from neural_astar.planner.astar import VanillaAstar
from neural_astar.utils.geometric_losses import (
    CombinedGeodesicLoss,
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
        direct_geo_supervision: bool = False,
    ):
        super().__init__()
        self.planner = planner
        self.vanilla_astar = VanillaAstar()
        self.config = config
        self.direct_geo_supervision = direct_geo_supervision

        if self.direct_geo_supervision:
            # Direct supervision: cost map + vector field
            self.geo_loss_weight = getattr(config.params, "geo_loss_weight", 1.0)
            self.geo_loss_fn = CombinedGeodesicLoss(
                dist_weight=getattr(config.params, "dist_loss_weight", 1.0),
                vec_weight=getattr(config.params, "vec_loss_weight", 1.0),
                consistency_weight=getattr(config.params, "consistency_weight", 1.0),
                warmup_epochs=getattr(config.params, "geo_warmup_epochs", 0),
                cons_warmup_epochs=getattr(config.params, "cons_warmup_epochs", 0),
                eikonal_weight=getattr(config.params, "eikonal_weight", 0.0),
            )

        # Optimize with torch.compile (PyTorch 2.0+)
        # "reduce-overhead" is beneficial for small batches and loops (like Star)
        if hasattr(torch, "compile") and sys.platform != "win32":
            # Windows support for torch.compile is experimental, skipping on Windows to avoid Triton errors
            try:
                self.planner = torch.compile(self.planner, mode="reduce-overhead")
            except Exception:
                self.planner = torch.compile(self.planner) # Fallback to default


    def forward(self, map_designs, start_maps, goal_maps):
        return self.planner(map_designs, start_maps, goal_maps)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer batch to device with non-blocking for async CPU-GPU transfer."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(
                self.transfer_batch_to_device(item, device, dataloader_idx)
                for item in batch
            )
        elif isinstance(batch, dict):
            return {
                key: self.transfer_batch_to_device(val, device, dataloader_idx)
                for key, val in batch.items()
            }
        return batch

    def configure_optimizers(self):
        if not self.direct_geo_supervision:
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
                    "frequency": 1,
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
        if self.direct_geo_supervision:
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
            path_weight = getattr(self.config.params, "path_loss_weight", 1.0)
            path_loss = path_loss * path_weight

            # Geo supervision uses auxiliary geodesic predictions when available.
            pred_dist = pred_vx = pred_vy = None
            
            # Attributes might be hidden under _orig_mod if compiled
            planner_module = self.planner._orig_mod if hasattr(self.planner, "_orig_mod") else self.planner
            
            geo_pred = (
                planner_module.get_geo_predictions()
                if hasattr(planner_module, "get_geo_predictions")
                else None
            )
            if isinstance(geo_pred, dict):
                pred_dist = geo_pred.get("dist", geo_pred.get("distance"))
                pred_vx = geo_pred.get("vx")
                pred_vy = geo_pred.get("vy")

            # Backward-compatible fallback (older "direct" encoders used cost map as distance)
            if pred_dist is None and hasattr(planner_module, "get_cost_map"):
                pred_dist = planner_module.get_cost_map()

            if (pred_vx is None or pred_vy is None) and hasattr(
                planner_module, "get_vector_field"
            ):
                vector_field = planner_module.get_vector_field()
                if vector_field is not None and vector_field[0] is not None:
                    pred_vx, pred_vy = vector_field

            geo_loss, loss_dict = self.geo_loss_fn(
                pred_dist,
                pred_vx,
                pred_vy,
                dist,
                vx,
                vy,
                reachable,
                current_epoch=self.current_epoch,
            )

            total_loss = path_loss + self.geo_loss_weight * geo_loss

            self.log(
                "metrics/train_loss",
                total_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("metrics/train_path_loss", path_loss)
            for k, v in loss_dict.items():
                name = f"metrics/train_{k.replace('/', '_')}"
                self.log(name, v, on_step=False, on_epoch=True)
            return total_loss

        map_designs, start_maps, goal_maps, opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = F.l1_loss(outputs.histories, opt_trajs)
        path_weight = getattr(self.config.params, "path_loss_weight", 1.0)
        loss = loss * path_weight
        self.log(
            "metrics/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.direct_geo_supervision:
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
            path_weight = getattr(self.config.params, "path_loss_weight", 1.0)
            path_loss = path_loss * path_weight

            pred_dist = pred_vx = pred_vy = None
            
            # Attributes might be hidden under _orig_mod if compiled
            planner_module = self.planner._orig_mod if hasattr(self.planner, "_orig_mod") else self.planner

            geo_pred = (
                planner_module.get_geo_predictions()
                if hasattr(planner_module, "get_geo_predictions")
                else None
            )
            if isinstance(geo_pred, dict):
                pred_dist = geo_pred.get("dist", geo_pred.get("distance"))
                pred_vx = geo_pred.get("vx")
                pred_vy = geo_pred.get("vy")

            if pred_dist is None and hasattr(planner_module, "get_cost_map"):
                pred_dist = planner_module.get_cost_map()

            if (pred_vx is None or pred_vy is None) and hasattr(
                planner_module, "get_vector_field"
            ):
                vector_field = planner_module.get_vector_field()
                if vector_field is not None and vector_field[0] is not None:
                    pred_vx, pred_vy = vector_field

            geo_loss, loss_dict = self.geo_loss_fn(
                pred_dist,
                pred_vx,
                pred_vy,
                dist,
                vx,
                vy,
                reachable,
                current_epoch=self.current_epoch,
            )
            loss = path_loss + self.geo_loss_weight * geo_loss

            self.log(
                "metrics/val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("metrics/val_path_loss", path_loss)
            for k, v in loss_dict.items():
                name = f"metrics/val_{k.replace('/', '_')}"
                self.log(name, v, on_step=False, on_epoch=True)
            vanilla_map = map_designs
        else:
            map_designs, start_maps, goal_maps, opt_trajs = val_batch
            outputs = self.forward(map_designs, start_maps, goal_maps)
            loss = F.l1_loss(outputs.histories, opt_trajs)
            path_weight = getattr(self.config.params, "path_loss_weight", 1.0)
            loss = loss * path_weight
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
        torch.backends.cudnn.deterministic = False # Faster, less reproducible
        torch.backends.cudnn.benchmark = True # Optimized for fixed input size

    np.random.seed(seed)
    random.seed(seed)
