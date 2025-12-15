# heat_method/solver.py
from __future__ import annotations

from typing import Tuple, Optional
import warnings

import numpy as np
import potpourri3d as pp3d
from numpy.typing import NDArray
from scipy import ndimage

from .dtypes import (
    GridMap,
    VectorField,
    GeodesicGuidance,
    MeshData,
)



# Constants

# Relative tolerance for degenerate triangle detection
# Triangles with area < DEGENERATE_AREA_REL_TOL * max_area are considered degenerate
_DEGENERATE_AREA_REL_TOL: float = 1e-12

# Minimum absolute area threshold (fallback when max_area is very small)
_DEGENERATE_AREA_ABS_TOL: float = 1e-20

# Tolerance for vector magnitude during normalization
_VECTOR_NORM_TOL: float = 1e-10

# Default upscaling factor
_DEFAULT_UPSCALE: int = 2


class GeodesicHeatSolver:
    """Heat Method based Geodesic Distance Solver."""

    @staticmethod
    def _grid_to_mesh_with_mapping(binary_map: GridMap) -> MeshData:
        """Convert a binary grid to triangle mesh with bidirectional mapping."""
        H, W = binary_map.shape
        valid_mask = binary_map == 1
        num_vertices = int(np.sum(valid_mask))
        
        # Handle empty map
        if num_vertices == 0:
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                vertex_ids=np.full((H, W), -1, dtype=np.int32),
                face_pixels=np.zeros((0, 2), dtype=np.int32)
            )
        
        # Create vertex ID grid (-1 for obstacles/empty)
        vertex_ids = np.full((H, W), -1, dtype=np.int32)
        vertex_ids[valid_mask] = np.arange(num_vertices, dtype=np.int32)
        
        # Create vertex positions (x=col, y=row, z=0)
        rows, cols = np.nonzero(valid_mask)
        vertices = np.column_stack([
            cols.astype(np.float64),  # x = column
            rows.astype(np.float64),  # y = row
            np.zeros(num_vertices, dtype=np.float64)  # z = 0 (planar)
        ])
        
        # Get vertex IDs for all 2×2 block corners
        tl = vertex_ids[:-1, :-1]  # Top-left
        tr = vertex_ids[:-1, 1:]   # Top-right
        bl = vertex_ids[1:, :-1]   # Bottom-left
        br = vertex_ids[1:, 1:]    # Bottom-right
        
        # Block pixel coordinates (row, col of TL corner)
        block_rows, block_cols = np.meshgrid(
            np.arange(H - 1), np.arange(W - 1), indexing='ij'
        )
        
        # Triangle validity masks (all 3 vertices must exist)
        # Triangle 1: TL-BL-TR (CCW when viewed from +Z)
        mask_t1 = (tl >= 0) & (bl >= 0) & (tr >= 0)
        # Triangle 2: BL-BR-TR (CCW when viewed from +Z)
        mask_t2 = (bl >= 0) & (br >= 0) & (tr >= 0)
        
        num_t1 = int(np.sum(mask_t1))
        num_t2 = int(np.sum(mask_t2))
        total_faces = num_t1 + num_t2
        
        # Handle case with no valid triangles
        if total_faces == 0:
            return MeshData(
                vertices=vertices,
                faces=np.zeros((0, 3), dtype=np.int32),
                vertex_ids=vertex_ids,
                face_pixels=np.zeros((0, 2), dtype=np.int32)
            )
        
        # Pre-allocate face and pixel arrays
        faces = np.empty((total_faces, 3), dtype=np.int32)
        face_pixels = np.empty((total_faces, 2), dtype=np.int32)
        
        # Fill Triangle 1 data
        if num_t1 > 0:
            faces[:num_t1, 0] = tl[mask_t1]
            faces[:num_t1, 1] = bl[mask_t1]
            faces[:num_t1, 2] = tr[mask_t1]
            face_pixels[:num_t1, 0] = block_rows[mask_t1]
            face_pixels[:num_t1, 1] = block_cols[mask_t1]
        
        # Fill Triangle 2 data
        if num_t2 > 0:
            faces[num_t1:, 0] = bl[mask_t2]
            faces[num_t1:, 1] = br[mask_t2]
            faces[num_t1:, 2] = tr[mask_t2]
            face_pixels[num_t1:, 0] = block_rows[mask_t2]
            face_pixels[num_t1:, 1] = block_cols[mask_t2]
        
        # Remove unreferenced vertices (isolated pixels without triangles)
        referenced = np.unique(faces.flatten())
        if len(referenced) < len(vertices):
            # Create old→new index mapping
            old_to_new = np.full(len(vertices), -1, dtype=np.int32)
            old_to_new[referenced] = np.arange(len(referenced), dtype=np.int32)
            
            # Update vertices and faces
            vertices = vertices[referenced]
            faces = old_to_new[faces]
            
            # Update vertex_ids grid
            valid_vertex_mask = vertex_ids >= 0
            vertex_ids = np.where(
                valid_vertex_mask,
                old_to_new[vertex_ids],
                -1
            ).astype(np.int32)
        
        return MeshData(vertices, faces, vertex_ids, face_pixels)
    
    
    @staticmethod
    def _compute_face_gradients(
        V: NDArray[np.float64],
        F: NDArray[np.int32],
        scalar_field: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """Compute gradient of scalar field on triangle faces."""
        # Vertex positions for each face
        v0 = V[F[:, 0]]  # (M, 3)
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        
        # Scalar values at face vertices
        u0 = scalar_field[F[:, 0]]  # (M,)
        u1 = scalar_field[F[:, 1]]
        u2 = scalar_field[F[:, 2]]
        
        # Edge vectors (opposite to each vertex)
        # e0 = v2 - v1 (opposite to v0)
        # e1 = v0 - v2 (opposite to v1)
        # e2 = v1 - v0 (opposite to v2)
        e0 = v2 - v1
        e1 = v0 - v2
        e2 = v1 - v0
        
        # Compute signed area × 2 via cross product z-component
        # For XY-plane triangles: (v1-v0) × (v2-v0) = (0, 0, 2*signed_area)
        double_area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
                      (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        
        # === ADAPTIVE DEGENERATE DETECTION ===
        # Use relative threshold based on maximum area in the mesh
        abs_double_area = np.abs(double_area)
        max_area = np.max(abs_double_area)
        
        # Threshold: max of relative and absolute tolerances
        area_threshold = max(
            _DEGENERATE_AREA_REL_TOL * max_area,
            _DEGENERATE_AREA_ABS_TOL
        )
        valid = abs_double_area > area_threshold
        
        # Safe division denominator
        double_area_safe = np.where(valid, double_area, 1.0)
        
        # Determine face orientation (+1 for CCW, -1 for CW when viewed from +Z)
        orientation = np.sign(double_area)
        orientation_safe = np.where(valid, orientation, 1.0)
        
        # Rotate edges 90° CCW: (ex, ey) → (-ey, ex)
        # Multiply by orientation to handle both winding orders
        rot_e0_x = -e0[:, 1] * orientation_safe
        rot_e0_y = e0[:, 0] * orientation_safe
        rot_e1_x = -e1[:, 1] * orientation_safe
        rot_e1_y = e1[:, 0] * orientation_safe
        rot_e2_x = -e2[:, 1] * orientation_safe
        rot_e2_y = e2[:, 0] * orientation_safe
        
        # Weighted sum: ∇u = Σᵢ uᵢ (N × eᵢ) / 2A
        grad_x = (u0 * rot_e0_x + u1 * rot_e1_x + u2 * rot_e2_x) / double_area_safe
        grad_y = (u0 * rot_e0_y + u1 * rot_e1_y + u2 * rot_e2_y) / double_area_safe
        
        # Zero out degenerate faces
        grad_x = np.where(valid, grad_x, 0.0)
        grad_y = np.where(valid, grad_y, 0.0)
        
        return np.column_stack([grad_x, grad_y]).astype(np.float32)

    
    @staticmethod
    def _rasterize_face_gradients(
        face_grads: NDArray[np.float32],
        face_pixels: NDArray[np.int32],
        H: int,
        W: int,
        reachable_mask: NDArray[np.bool_],
    ) -> Tuple[VectorField, VectorField]:
        """Rasterize face gradients to pixel grid."""
        # Accumulator arrays (use float64 for precision during accumulation)
        grad_sum_x = np.zeros((H, W), dtype=np.float64)
        grad_sum_y = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)
        
        # Extract face data
        rows = face_pixels[:, 0]
        cols = face_pixels[:, 1]
        gx = face_grads[:, 0].astype(np.float64)
        gy = face_grads[:, 1].astype(np.float64)
        
        # Each face contributes to 4 pixels: (r,c), (r,c+1), (r+1,c), (r+1,c+1)
        M = len(rows)
        
        # Create indices for all 4 pixel positions per face
        dr = np.array([0, 0, 1, 1])
        dc = np.array([0, 1, 0, 1])
        
        all_r = (rows[:, None] + dr).ravel()  # (M*4,)
        all_c = (cols[:, None] + dc).ravel()  # (M*4,)
        all_gx = np.tile(gx, 4)
        all_gy = np.tile(gy, 4)
        all_ones = np.ones(M * 4, dtype=np.float64)
        
        # Bounds check
        valid_idx = (all_r >= 0) & (all_r < H) & (all_c >= 0) & (all_c < W)
        
        # Scatter add
        np.add.at(grad_sum_x, (all_r[valid_idx], all_c[valid_idx]), all_gx[valid_idx])
        np.add.at(grad_sum_y, (all_r[valid_idx], all_c[valid_idx]), all_gy[valid_idx])
        np.add.at(count, (all_r[valid_idx], all_c[valid_idx]), all_ones[valid_idx])
        
        # Compute average
        valid_count = count > 0
        vec_x = np.zeros((H, W), dtype=np.float32)
        vec_y = np.zeros((H, W), dtype=np.float32)
        
        vec_x[valid_count] = (grad_sum_x[valid_count] / count[valid_count]).astype(np.float32)
        vec_y[valid_count] = (grad_sum_y[valid_count] / count[valid_count]).astype(np.float32)
        
        # Negate: gradient points uphill (away from goal), we want downhill (toward goal)
        vec_x = -vec_x
        vec_y = -vec_y
        
        # Mask unreachable regions
        vec_x[~reachable_mask] = 0.0
        vec_y[~reachable_mask] = 0.0
        
        # Normalize to unit vectors
        magnitude = np.sqrt(vec_x**2 + vec_y**2)
        valid_mag = magnitude > _VECTOR_NORM_TOL
        
        vec_x[valid_mag] /= magnitude[valid_mag]
        vec_y[valid_mag] /= magnitude[valid_mag]
        
        # Zero out vectors with insufficient magnitude
        vec_x[~valid_mag] = 0.0
        vec_y[~valid_mag] = 0.0
        
        return vec_x, vec_y
    
    
    @staticmethod
    def _find_reachable_region(
        grid_map: GridMap,
        goal_pos: Tuple[int, int],
    ) -> Tuple[NDArray[np.bool_], Tuple[int, int]]:
        """Extract connected region reachable from goal."""
        H, W = grid_map.shape
        g_r, g_c = goal_pos
        
        # Clamp goal to grid bounds
        g_r = int(np.clip(g_r, 0, H - 1))
        g_c = int(np.clip(g_c, 0, W - 1))
        
        # 8-connectivity structure (Moore neighborhood)
        structure = np.ones((3, 3), dtype=np.int32)
        labeled, num_labels = ndimage.label(grid_map, structure=structure)
        
        # Handle goal on obstacle: find nearest passable cell
        if grid_map[g_r, g_c] == 0:
            valid_cells = np.argwhere(grid_map == 1)
            if len(valid_cells) == 0:
                # No passable cells at all
                return np.zeros((H, W), dtype=bool), (g_r, g_c)
            
            # Find nearest passable cell (Euclidean distance)
            dists_sq = np.sum((valid_cells - np.array([g_r, g_c]))**2, axis=1)
            nearest_idx = np.argmin(dists_sq)
            g_r, g_c = valid_cells[nearest_idx]
        
        # Get label of goal's connected component
        label = labeled[g_r, g_c]
        if label == 0:
            # Goal is somehow still not in a labeled region
            return np.zeros((H, W), dtype=bool), (int(g_r), int(g_c))
        
        return (labeled == label), (int(g_r), int(g_c))

    
    @classmethod
    def compute_guidance(
        cls,
        grid_map: GridMap,
        goal_pos: Tuple[int, int],
        t_coef: float = 1.0,
        use_robust: bool = True,
        upscale_factor: int = _DEFAULT_UPSCALE,
    ) -> GeodesicGuidance:
        """Compute geodesic guidance field using the Heat Method. """
        H, W = grid_map.shape
        
        # 1. Reachability analysis
        reachable_mask, (g_r, g_c) = cls._find_reachable_region(grid_map, goal_pos)
        
        if not np.any(reachable_mask):
            return cls._empty_result(H, W)
        
        # 2. Upscaling for robust topology
        # 1-pixel corridors may fail to form triangles in 2×2 block triangulation
        scale = upscale_factor
        effective_map = (grid_map * reachable_mask).astype(np.float32)
        effective_map_up = effective_map.repeat(scale, axis=0).repeat(scale, axis=1)
        
        # Scale goal position
        g_r_up, g_c_up = g_r * scale, g_c * scale
        H_up, W_up = effective_map_up.shape
        
        # 3. Mesh construction
        mesh = cls._grid_to_mesh_with_mapping(effective_map_up)
        
        if len(mesh.faces) == 0:
            return cls._empty_result(H, W)
        
        # Find goal vertex
        goal_idx = mesh.vertex_ids[g_r_up, g_c_up]
        if goal_idx == -1:
            # Goal position has no corresponding vertex
            return cls._empty_result(H, W)
        
        # 4. Heat Method distance computation
        try:
            # potpourri3d internally computes: t = t_coef × (mean_edge_length)²
            # Since upscaling halves edge length, t automatically scales by 1/4
            solver = pp3d.MeshHeatMethodDistanceSolver(
                mesh.vertices,
                mesh.faces,
                t_coef=t_coef,
                use_robust=use_robust,
            )
            vertex_dists = solver.compute_distance(goal_idx)
        except Exception as e:
            warnings.warn(f"[GeodesicHeatSolver] Heat method failed: {e}")
            return cls._empty_result(H, W)
        
        # 5. Map vertex distances to upscaled grid
        dist_grid_up = np.full((H_up, W_up), np.nan, dtype=np.float64)
        valid_verts = mesh.vertex_ids >= 0
        dist_grid_up[valid_verts] = vertex_dists[mesh.vertex_ids[valid_verts]]
        
        # 6. Compute face gradients
        face_grads = cls._compute_face_gradients(
            mesh.vertices, mesh.faces, vertex_dists
        )
        
        # 7. Rasterize gradients to upscaled grid
        reachable_up = effective_map_up > 0.5
        vec_x_up, vec_y_up = cls._rasterize_face_gradients(
            face_grads, mesh.face_pixels, H_up, W_up, reachable_up
        )
        
        # 8. Downscale to original resolution
        # Use subsampling (fast, works well for uniform grids)
        dist_grid = dist_grid_up[::scale, ::scale]
        vec_x = vec_x_up[::scale, ::scale].copy()
        vec_y = vec_y_up[::scale, ::scale].copy()
        
        # 9. Normalize distance by map diagonal
        diagonal = np.sqrt(H**2 + W**2)
        dist_norm = np.ones((H, W), dtype=np.float32)
        valid_d = np.isfinite(dist_grid)
        dist_norm[valid_d] = (dist_grid[valid_d] / diagonal).astype(np.float32)
        
        # 10. Ensure vectors are zero in unreachable areas
        vec_x[~reachable_mask] = 0.0
        vec_y[~reachable_mask] = 0.0
        
        # 11. Re-normalize vectors (subsampling may have broken unit length)
        magnitude = np.sqrt(vec_x**2 + vec_y**2)
        valid_mag = magnitude > _VECTOR_NORM_TOL
        
        vec_x[valid_mag] /= magnitude[valid_mag]
        vec_y[valid_mag] /= magnitude[valid_mag]
        vec_x[~valid_mag] = 0.0
        vec_y[~valid_mag] = 0.0
        
        return GeodesicGuidance(
            vec_x=vec_x,
            vec_y=vec_y,
            dist_normalized=dist_norm,
            reachable=reachable_mask.astype(np.float32),
        )
    
    @staticmethod
    def _empty_result(H: int, W: int) -> GeodesicGuidance:
        """Create empty result for unreachable/invalid cases. """
        return GeodesicGuidance(
            vec_x=np.zeros((H, W), dtype=np.float32),
            vec_y=np.zeros((H, W), dtype=np.float32),
            dist_normalized=np.ones((H, W), dtype=np.float32),
            reachable=np.zeros((H, W), dtype=np.float32),
        )