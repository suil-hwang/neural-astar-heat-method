# src/neural_astar/planner/astar.py
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from . import encoder
from .differentiable_astar import AstarOutput, DifferentiableAstar
from .pq_astar import pq_astar


class VanillaAstar(nn.Module):
    def __init__(
        self,
        g_ratio: float = 0.5,
        use_differentiable_astar: bool = True,
    ):
        """Vanilla A* search."""
        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=1.0,
        )
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def perform_astar(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        obstacles_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:

        astar = (
            self.astar
            if self.use_differentiable_astar
            else partial(pq_astar, g_ratio=self.g_ratio)
        )

        astar_outputs = astar(
            map_designs,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )

        return astar_outputs

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        """Perform A* search """
        cost_maps = map_designs
        obstacles_maps = map_designs

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )


class NeuralAstar(VanillaAstar):
    def __init__(
        self,
        g_ratio: float = 0.5,
        Tmax: float = 1.0,
        encoder_input: str = "m+",
        encoder_arch: str = "CNN",
        encoder_depth: int = 4,
        learn_obstacles: bool = False,
        const: float = None,
        use_differentiable_astar: bool = True,
        encoder_kwargs: dict[str, object] | None = None,
    ):
        """
        Neural A* search.
        
        For Gated Fusion:
        - Set encoder_arch="GatedUnet" or "GatedCNN"
        - Set encoder_input="mgsvxyr" (7ch) or "msgdr" (5ch)
        - Visualize gate: planner.get_gate_map()
        """
        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=Tmax,
        )
        self.encoder_input = encoder_input
        self.encoder_arch = encoder_arch
        
        # Create encoder
        encoder_arch_cls = getattr(encoder, encoder_arch)
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = encoder_arch_cls(
            len(self.encoder_input),
            encoder_depth,
            const,
            **encoder_kwargs,
        )
        self._last_vector_field = None
        self._last_cost_map = None
        self._last_geo_predictions = None
        
        self.learn_obstacles = learn_obstacles
        if self.learn_obstacles:
            print("WARNING: learn_obstacles has been set to True")
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def encode(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
    ) -> torch.tensor:
        """
        Encode input to cost map.
        
        For Gated Fusion (7-channel input):
        - map_designs is pre-concatenated [map, start, goal, vx, vy, dist, reachable]
        - Encoder internally splits into Base(3ch) / Heat(4ch)
        """
        n_channels = len(self.encoder_input)

        if "+" in self.encoder_input:
            # mode=neural_astar: map + (start + goal)
            inputs = map_designs
            if map_designs.shape[-1] == start_maps.shape[-1]:
                inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
            else:
                upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
                inputs = torch.cat((inputs, upsampler(start_maps + goal_maps)), dim=1)
        elif "msg" in self.encoder_input.lower():
            # Explicit 3-channel map/start/goal
            if map_designs.shape[-2:] != start_maps.shape[-2:]:
                upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
                start_maps = upsampler(start_maps)
                goal_maps = upsampler(goal_maps)
            inputs = torch.cat((map_designs[:, :1], start_maps, goal_maps), dim=1)
        else:
            inputs = map_designs

        encoded = self.encoder(inputs)
        self._last_vector_field = None
        self._last_geo_predictions = None

        if isinstance(encoded, dict):
            cost_maps = encoded.get("cost_map", encoded.get("cost_maps", None))
            self._last_vector_field = encoded.get("vector_field", None)
            self._last_geo_predictions = encoded.get("geo_predictions", None)
            if cost_maps is None:
                raise ValueError("Encoder dict output must contain 'cost_map'")
        else:
            cost_maps = encoded

        if hasattr(self.encoder, "get_geo_predictions"):
            self._last_geo_predictions = self.encoder.get_geo_predictions()

        if hasattr(self.encoder, "get_vector_field"):
            self._last_vector_field = self.encoder.get_vector_field()

        self._last_cost_map = cost_maps
        return cost_maps
    
    def get_gate_map(self) -> torch.Tensor | None:
        """Return Gate Map (for Gated encoders)."""
        if hasattr(self.encoder, 'get_gate_map'):
            return self.encoder.get_gate_map()
        return None

    def get_attention_weights(self):
        if hasattr(self.encoder, "get_attention_weights"):
            return self.encoder.get_attention_weights()
        return None

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        """Perform neural A* search."""
        cost_maps = self.encode(map_designs, start_maps, goal_maps)
        
        # Extract obstacle map
        n_channels = len(self.encoder_input)
        if not self.learn_obstacles:
            obstacles_maps = map_designs[:, :1]
        else:
            obstacles_maps = torch.ones_like(start_maps[:, :1])

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )

    def get_vector_field(self):
        return self._last_vector_field

    def get_geo_predictions(self):
        """Return last geodesic predictions for auxiliary supervision (if available)."""
        return self._last_geo_predictions

    def get_cost_map(self) -> torch.Tensor | None:
        """Return the last predicted cost map used by A*."""
        return self._last_cost_map
