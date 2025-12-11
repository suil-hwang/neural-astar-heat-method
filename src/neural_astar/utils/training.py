# src/neural_astar/utils/training.py
from __future__ import annotations

import logging
import random
import re
from glob import glob
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    PyTorch Lightning module with geodesic-aware attention support.
    """

    def __init__(
        self,
        planner,
        config,
        use_guidance: bool = False,
        geo_supervision: bool = False,
        use_phi: bool = False,
        num_anchors: int = 8,
    ):
        super().__init__()
        self.planner = planner
        self.vanilla_astar = VanillaAstar()
        self.config = config
        self.geo_supervision = geo_supervision
        self.use_phi = use_phi
        self.num_anchors = num_anchors

        self.geo_loss_weight = getattr(config.params, "geo_loss_weight", 0.1)
        self.geo_warmup_epochs = getattr(config.params, "geo_warmup_epochs", 5)
        self.vec_loss_weight = getattr(config.params, "vec_loss_weight", 0.1)

        self.beta_warmup_epochs = getattr(config.params, "beta_warmup_epochs", 10)
        self.beta_unfreeze_epoch = getattr(config.params, "beta_unfreeze_epoch", 5)

        self.geo_loss_decay_epoch = getattr(config.params, "geo_loss_decay_epoch", 15)
        self.geo_loss_decay_factor = getattr(config.params, "geo_loss_decay_factor", 0.5)

        temps = getattr(config.params, "geo_temps", [0.5, 1.0, 2.0])
        self.geo_loss_fn = GoalCentricGeodesicLoss(temperatures=list(temps))
        self.vec_loss_fn = VectorFieldLoss(loss_type="cosine")

        self.save_hyperparameters(ignore=["planner", "config"])
        self._beta_frozen = False
        if self.geo_supervision:
            self._beta_frozen = self._set_beta_requires_grad(False)

    def forward(
        self,
        map_designs,
        start_maps,
        goal_maps,
        phi: Optional[torch.Tensor] = None,
        reachable: Optional[torch.Tensor] = None,
    ):
        encoder = self.planner.encoder

        if hasattr(encoder, "forward") and phi is not None:
            import inspect

            sig = inspect.signature(encoder.forward)
            params = list(sig.parameters.keys())

            if "phi" in params:
                cost_maps = encoder(
                    self._prepare_base_input(map_designs, start_maps, goal_maps),
                    phi=phi,
                    spatial_mask=reachable,
                )
            else:
                cost_maps = self.planner.encode(map_designs, start_maps, goal_maps)
        else:
            cost_maps = self.planner.encode(map_designs, start_maps, goal_maps)

        obstacles_maps = map_designs[:, :1]
        return self.planner.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results=False,
        )

    def _prepare_base_input(self, map_designs, start_maps, goal_maps):
        if map_designs.shape[-2:] != start_maps.shape[-2:]:
            upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
            start_maps = upsampler(start_maps)
            goal_maps = upsampler(goal_maps)
        return torch.cat([map_designs[:, :1], start_maps, goal_maps], dim=1)

    def configure_optimizers(self):
        if not self.geo_supervision:
            return torch.optim.RMSprop(
                self.planner.parameters(),
                self.config.params.lr,
            )

        optimizer = torch.optim.AdamW(
            self.planner.parameters(),
            lr=self.config.params.lr,
            weight_decay=0.01,
        )

        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.params.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        except (RuntimeError, MisconfigurationException) as exc:
            logging.warning(
                f"estimated_stepping_batches unavailable ({exc}). "
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
                },
            }

    def on_train_epoch_start(self):
        epoch = self.current_epoch

        if self._beta_frozen and epoch >= self.beta_unfreeze_epoch:
            if self._set_beta_requires_grad(True):
                self._beta_frozen = False
                logging.info(f"Epoch {epoch}: Unfreezing beta parameters")

        if hasattr(self.planner.encoder, "get_betas"):
            betas = self.planner.encoder.get_betas()
            if betas:
                for idx, beta in enumerate(betas):
                    self.log(f"params/beta_{idx}", beta.mean().item())

    def _set_beta_requires_grad(self, requires_grad: bool):
        changed = False
        for name, param in self.planner.named_parameters():
            if "beta" in name.lower():
                param.requires_grad = requires_grad
                changed = True
        return changed

    def _get_warmup_factor(self) -> float:
        if self.geo_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self.current_epoch) / float(self.geo_warmup_epochs))

    def _get_geo_loss_weight(self) -> float:
        base_weight = self.geo_loss_weight * self._get_warmup_factor()
        if self.current_epoch >= self.geo_loss_decay_epoch:
            base_weight *= self.geo_loss_decay_factor
        return base_weight

    def _unpack_batch(self, batch) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}

        result["map"] = batch[0]
        result["start"] = batch[1]
        result["goal"] = batch[2]
        result["traj"] = batch[3]

        batch_len = len(batch)
        if batch_len == 5:
            result["phi"] = batch[4]
        elif batch_len == 8:
            result["dist"] = batch[4]
            result["reachable"] = batch[5]
            result["vx"] = batch[6]
            result["vy"] = batch[7]
        elif batch_len == 9:
            result["dist"] = batch[4]
            result["reachable"] = batch[5]
            result["vx"] = batch[6]
            result["vy"] = batch[7]
            result["phi"] = batch[8]
        return result

    def training_step(self, train_batch, batch_idx):
        batch_dict = self._unpack_batch(train_batch)

        map_designs = batch_dict["map"]
        start_maps = batch_dict["start"]
        goal_maps = batch_dict["goal"]
        opt_trajs = batch_dict["traj"]

        phi = batch_dict.get("phi")
        reachable = batch_dict.get("reachable")

        outputs = self.forward(
            map_designs,
            start_maps,
            goal_maps,
            phi=phi,
            reachable=reachable,
        )

        path_loss = F.l1_loss(outputs.histories, opt_trajs)

        geo_loss = torch.tensor(0.0, device=self.device)
        if self.geo_supervision and "dist" in batch_dict:
            attn_weights = self.planner.get_attention_weights()
            if attn_weights:
                geo_loss = self.geo_loss_fn(
                    attn_weights[-1],
                    batch_dict["dist"],
                    batch_dict.get("reachable"),
                )

        vec_loss = torch.tensor(0.0, device=self.device)
        if self.geo_supervision and "vx" in batch_dict and "vy" in batch_dict:
            vector_field = self.planner.get_vector_field()
            if vector_field is not None and vector_field[0] is not None:
                pred_vx, pred_vy = vector_field
                vec_loss = self.vec_loss_fn(
                    pred_vx,
                    pred_vy,
                    batch_dict["vx"],
                    batch_dict["vy"],
                    batch_dict.get("reachable", torch.ones_like(pred_vx)),
                )

        warmup_factor = self._get_warmup_factor()
        effective_geo_weight = self._get_geo_loss_weight()
        effective_vec_weight = self.vec_loss_weight * warmup_factor

        total_loss = (
            path_loss
            + effective_geo_weight * geo_loss
            + effective_vec_weight * vec_loss
        )

        self.log("metrics/train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("metrics/train_path_loss", path_loss)

        if isinstance(geo_loss, torch.Tensor) and geo_loss.item() > 0:
            self.log("metrics/train_geo_loss", geo_loss)

        if isinstance(vec_loss, torch.Tensor) and vec_loss.item() > 0:
            self.log("metrics/train_vec_loss", vec_loss)

        self.log("schedule/warmup_factor", warmup_factor)
        self.log("schedule/geo_weight", effective_geo_weight)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        batch_dict = self._unpack_batch(val_batch)

        map_designs = batch_dict["map"]
        start_maps = batch_dict["start"]
        goal_maps = batch_dict["goal"]
        opt_trajs = batch_dict["traj"]

        phi = batch_dict.get("phi")
        reachable = batch_dict.get("reachable")

        outputs = self.forward(
            map_designs,
            start_maps,
            goal_maps,
            phi=phi,
            reachable=reachable,
        )

        path_loss = F.l1_loss(outputs.histories, opt_trajs)

        geo_loss = torch.tensor(0.0, device=self.device)
        if self.geo_supervision and "dist" in batch_dict:
            attn_weights = self.planner.get_attention_weights()
            if attn_weights:
                geo_loss = self.geo_loss_fn(
                    attn_weights[-1],
                    batch_dict["dist"],
                    batch_dict.get("reachable"),
                )

        vec_loss = torch.tensor(0.0, device=self.device)
        if self.geo_supervision and "vx" in batch_dict and "vy" in batch_dict:
            vector_field = self.planner.get_vector_field()
            if vector_field is not None and vector_field[0] is not None:
                pred_vx, pred_vy = vector_field
                vec_loss = self.vec_loss_fn(
                    pred_vx,
                    pred_vy,
                    batch_dict["vx"],
                    batch_dict["vy"],
                    batch_dict.get("reachable", torch.ones_like(pred_vx)),
                )

        total_loss = path_loss + self.geo_loss_weight * geo_loss + self.vec_loss_weight * vec_loss

        self.log("metrics/val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("metrics/val_path_loss", path_loss)

        if isinstance(geo_loss, torch.Tensor) and geo_loss.item() > 0:
            self.log("metrics/val_geo_loss", geo_loss)

        if isinstance(vec_loss, torch.Tensor) and vec_loss.item() > 0:
            self.log("metrics/val_vec_loss", vec_loss)

        vanilla_map = map_designs[:, :1] if map_designs.shape[1] > 1 else map_designs

        if vanilla_map.shape[1] == 1:
            va_outputs = self.vanilla_astar(vanilla_map, start_maps, goal_maps)

            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()

            exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            p_exp = np.maximum((exp_astar - exp_na) / (exp_astar + 1e-8), 0.0).mean()

            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

            self.log("metrics/p_opt", p_opt)
            self.log("metrics/p_exp", p_exp)
            self.log("metrics/h_mean", h_mean)

        return total_loss


def set_global_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


from dataclasses import dataclass


@dataclass
class GeoAttentionConfig:
    """Configuration for geodesic attention training."""

    lr: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 32

    geo_loss_weight: float = 0.1
    geo_warmup_epochs: int = 5
    geo_temps: Tuple[float, ...] = (0.5, 1.0, 2.0)

    vec_loss_weight: float = 0.1

    beta_warmup_epochs: int = 10
    beta_unfreeze_epoch: int = 5

    geo_loss_decay_epoch: int = 15
    geo_loss_decay_factor: float = 0.5

    num_anchors: int = 8

    encoder_arch: str = "GeoAttentionUnet"
    encoder_depth: int = 4
    attention_heads: int = 4
    num_attention_blocks: int = 2
    init_beta: float = 0.5
    use_geodesic_pe: bool = True
