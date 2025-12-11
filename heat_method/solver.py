# heat_method/solver.py
from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
from scipy import ndimage
from typing import Tuple

from .dtypes import (
    GridMap,
    VectorField,
    GeodesicGuidance,
    MeshData,
)


class GeodesicHeatSolver:
    """Heat Method based Geodesic Distance Solver."""
    
    @staticmethod
    def _grid_to_mesh_with_mapping(binary_map: GridMap) -> MeshData:
        """Convert a grid to mesh while preserving Face→Pixel mapping."""
        H, W = binary_map.shape
        valid_mask = binary_map == 1
        num_vertices = np.sum(valid_mask)
        
        if num_vertices == 0:
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                vertex_ids=np.full((H, W), -1, dtype=np.int32),
                face_pixels=np.zeros((0, 2), dtype=np.int32)
            )
        
        # Vertex IDs
        vertex_ids = np.full((H, W), -1, dtype=np.int32)
        vertex_ids[valid_mask] = np.arange(num_vertices, dtype=np.int32)
        
        # Vertices (x=col, y=row, z=0)
        rows, cols = np.nonzero(valid_mask)
        vertices = np.column_stack([
            cols.astype(np.float64),
            rows.astype(np.float64),
            np.zeros(num_vertices, dtype=np.float64)
        ])
        
        # 2x2 block corners
        tl = vertex_ids[:-1, :-1]
        tr = vertex_ids[:-1, 1:]
        bl = vertex_ids[1:, :-1]
        br = vertex_ids[1:, 1:]
        
        # Block pixel coordinates (row, col of TL corner)
        block_rows, block_cols = np.meshgrid(
            np.arange(H - 1), np.arange(W - 1), indexing='ij'
        )
        
        # Triangle 1: TL-BL-TR (CCW)
        mask_t1 = (tl >= 0) & (bl >= 0) & (tr >= 0)
        # Triangle 2: BL-BR-TR (CCW)
        mask_t2 = (bl >= 0) & (br >= 0) & (tr >= 0)
        
        num_t1 = np.sum(mask_t1)
        num_t2 = np.sum(mask_t2)
        total_faces = num_t1 + num_t2
        
        if total_faces == 0:
            return MeshData(
                vertices=vertices,
                faces=np.zeros((0, 3), dtype=np.int32),
                vertex_ids=vertex_ids,
                face_pixels=np.zeros((0, 2), dtype=np.int32)
            )
        
        # Pre-allocate
        faces = np.empty((total_faces, 3), dtype=np.int32)
        face_pixels = np.empty((total_faces, 2), dtype=np.int32)
        
        # Triangle 1
        if num_t1 > 0:
            faces[:num_t1, 0] = tl[mask_t1]
            faces[:num_t1, 1] = bl[mask_t1]
            faces[:num_t1, 2] = tr[mask_t1]
            face_pixels[:num_t1, 0] = block_rows[mask_t1]
            face_pixels[:num_t1, 1] = block_cols[mask_t1]
        
        # Triangle 2
        if num_t2 > 0:
            faces[num_t1:, 0] = bl[mask_t2]
            faces[num_t1:, 1] = br[mask_t2]
            faces[num_t1:, 2] = tr[mask_t2]
            face_pixels[num_t1:, 0] = block_rows[mask_t2]
            face_pixels[num_t1:, 1] = block_cols[mask_t2]
        
        # Remove unreferenced vertices (isolated pixels that don't form triangles)
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
        V: np.ndarray, 
        F: np.ndarray, 
        scalar_field: np.ndarray
    ) -> np.ndarray:
        """Compute the gradient of a scalar field on triangular faces."""
        # Vertex positions for each face: (M, 3)
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        
        # Scalar values at face vertices: (M,)
        u0 = scalar_field[F[:, 0]]
        u1 = scalar_field[F[:, 1]]
        u2 = scalar_field[F[:, 2]]
        
        # Edge vectors (opposite to each vertex)
        # e0 = v2 - v1 (opposite to v0)
        # e1 = v0 - v2 (opposite to v1)
        # e2 = v1 - v0 (opposite to v2)
        e0 = v2 - v1
        e1 = v0 - v2
        e2 = v1 - v0
        
        # Face normal via cross product: n = (v1 - v0) × (v2 - v0)
        # For 2D grid embedded in XY plane, this gives (0, 0, 2*Area)
        cross = np.cross(v1 - v0, v2 - v0)  # (M, 3)
        
        # Signed area * 2 (z-component for XY-plane triangles)
        double_area = cross[:, 2]  # (M,)
        
        # Handle degenerate faces
        valid = np.abs(double_area) > 1e-10
        double_area_safe = np.where(valid, double_area, 1.0)
        
        # Unit normal (for XY plane, always (0, 0, 1) or (0, 0, -1))
        # But we need the actual normal for the cross product formula
        normal = np.zeros_like(cross)
        normal[valid, 2] = np.sign(double_area[valid])
        
        # Gradient contribution from each vertex: u_i * (n × e_i)
        # For n = (0, 0, 1) and e = (ex, ey, 0):
        #   n × e = (0*0 - 1*ey, 1*ex - 0*0, 0*ey - 0*ex) = (-ey, ex, 0)
        # This rotates the edge 90° CCW in the XY plane
        
        # Simplified for 2D (z=0):
        # (n × e)_x = -e_y * n_z
        # (n × e)_y = e_x * n_z
        
        n_z = normal[:, 2:3]  # (M, 1)
        
        # Rotate edges 90° (n × e)
        rot_e0 = np.column_stack([-e0[:, 1] * n_z.flatten(), e0[:, 0] * n_z.flatten()])
        rot_e1 = np.column_stack([-e1[:, 1] * n_z.flatten(), e1[:, 0] * n_z.flatten()])
        rot_e2 = np.column_stack([-e2[:, 1] * n_z.flatten(), e2[:, 0] * n_z.flatten()])
        
        # Sum weighted by scalar values
        grad = (u0[:, None] * rot_e0 + u1[:, None] * rot_e1 + u2[:, None] * rot_e2)
        
        # Divide by 2*Area
        grad = grad / double_area_safe[:, None]
        
        # Zero out degenerate faces
        grad[~valid] = 0
        
        # Flip gradient if signed_double_area is negative (CW winding order correction)
        flip_mask = double_area < 0
        grad[flip_mask] = -grad[flip_mask]
        
        return grad.astype(np.float32)  # (M, 2)

    
    @staticmethod
    def _rasterize_face_gradients(
        face_grads: np.ndarray,
        face_pixels: np.ndarray,
        H: int, 
        W: int,
        reachable_mask: np.ndarray
    ) -> Tuple[VectorField, VectorField]:
        """Rasterize face gradients onto the pixel grid."""
        # Accumulator arrays
        grad_sum_x = np.zeros((H, W), dtype=np.float64)
        grad_sum_y = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)
        
        # Each face contributes to its 2x2 block pixels
        # TL pixel is at (r, c), block covers (r:r+2, c:c+2)
        rows = face_pixels[:, 0]
        cols = face_pixels[:, 1]
        gx = face_grads[:, 0]
        gy = face_grads[:, 1]
        
        # Contribute to TL, TR, BL, BR pixels
        for dr in [0, 1]:
            for dc in [0, 1]:
                r = rows + dr
                c = cols + dc
                # Bounds check
                valid = (r < H) & (c < W)
                np.add.at(grad_sum_x, (r[valid], c[valid]), gx[valid])
                np.add.at(grad_sum_y, (r[valid], c[valid]), gy[valid])
                np.add.at(count, (r[valid], c[valid]), 1)
        
        # Average
        valid_count = count > 0
        vec_x = np.zeros((H, W), dtype=np.float32)
        vec_y = np.zeros((H, W), dtype=np.float32)
        
        vec_x[valid_count] = (grad_sum_x[valid_count] / count[valid_count]).astype(np.float32)
        vec_y[valid_count] = (grad_sum_y[valid_count] / count[valid_count]).astype(np.float32)
        
        # Negate: gradient points uphill, we want direction TO goal (downhill)
        vec_x = -vec_x
        vec_y = -vec_y
        
        # Mask unreachable
        vec_x[~reachable_mask] = 0
        vec_y[~reachable_mask] = 0
        
        # Normalize
        magnitude = np.sqrt(vec_x**2 + vec_y**2)
        valid_mag = magnitude > 1e-8
        vec_x[valid_mag] /= magnitude[valid_mag]
        vec_y[valid_mag] /= magnitude[valid_mag]
        
        return vec_x, vec_y
    
    @staticmethod
    def _find_reachable_region(
        grid_map: GridMap,
        goal_pos: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract the connected region reachable from the goal."""
        H, W = grid_map.shape
        g_r, g_c = goal_pos
        g_r = np.clip(g_r, 0, H - 1)
        g_c = np.clip(g_c, 0, W - 1)
        
        structure = np.ones((3, 3), dtype=np.int32)
        labeled, _ = ndimage.label(grid_map, structure=structure)
        
        if grid_map[g_r, g_c] == 0:
            valid = np.argwhere(grid_map == 1)
            if len(valid) == 0:
                return np.zeros((H, W), dtype=bool), (g_r, g_c)
            dists = np.sum((valid - [g_r, g_c])**2, axis=1)
            g_r, g_c = valid[np.argmin(dists)]
        
        label = labeled[g_r, g_c]
        if label == 0:
            return np.zeros((H, W), dtype=bool), (g_r, g_c)
        
        return (labeled == label), (int(g_r), int(g_c))
    
    @classmethod
    def compute_guidance(
        cls,
        grid_map: GridMap,
        goal_pos: Tuple[int, int],
        t_coef: float = 1.0,
        use_robust: bool = True,
    ) -> GeodesicGuidance:
        """Compute the Geodesic Guidance Field."""
        H, W = grid_map.shape
        
        # 1. Reachability analysis
        reachable_mask, (g_r, g_c) = cls._find_reachable_region(grid_map, goal_pos)
        if not np.any(reachable_mask):
            return cls._empty_result(H, W)
        
        # --- UPSCALING FOR ROBUST TOPOLOGY ---
        # 1-pixel wide corridors fail to form mesh faces (triangles) in 2x2 block analysis.
        # We upscale the map by 2x to ensure all corridors form valid faces.
        scale = 2
        effective_map = (grid_map * reachable_mask).astype(np.float32)
        effective_map_up = effective_map.repeat(scale, axis=0).repeat(scale, axis=1)
        
        # Scale goal position
        g_r_up, g_c_up = g_r * scale, g_c * scale
        
        # 2. Mesh construction (High Res)
        mesh = cls._grid_to_mesh_with_mapping(effective_map_up)
        
        if len(mesh.faces) == 0:
            return cls._empty_result(H, W)
        
        goal_idx = mesh.vertex_ids[g_r_up, g_c_up]
        if goal_idx == -1:
            return cls._empty_result(H, W)
        
        # 3. Heat Method distance (High Res)
        try:
            solver = pp3d.MeshHeatMethodDistanceSolver(
                mesh.vertices, mesh.faces, 
                t_coef=t_coef, use_robust=use_robust
            )
            vertex_dists = solver.compute_distance(goal_idx)
        except Exception as e:
            print(f"[GeodesicHeatSolver] Heat method failed: {e}")
            return cls._empty_result(H, W)
        
        # 4. Map distances to grid (High Res)
        H_up, W_up = effective_map_up.shape
        dist_grid_up = np.full((H_up, W_up), np.nan, dtype=np.float64)
        valid_verts = mesh.vertex_ids >= 0
        dist_grid_up[valid_verts] = vertex_dists[mesh.vertex_ids[valid_verts]]
        
        # 5. Compute gradients (High Res)
        face_grads = cls._compute_face_gradients(
            mesh.vertices, mesh.faces, vertex_dists
        )
        # Note: rasterize needs to know valid area of upscaled map
        vec_x_up, vec_y_up = cls._rasterize_face_gradients(
            face_grads, mesh.face_pixels, H_up, W_up, (effective_map_up > 0.5)
        )
        
        # --- DOWNSCALING RESULTS ---
        # Subsample to return to original resolution
        dist_grid = dist_grid_up[::scale, ::scale]
        vec_x = vec_x_up[::scale, ::scale]
        vec_y = vec_y_up[::scale, ::scale]
        
        # 6. Normalize distance
        diagonal = np.sqrt(H**2 + W**2)
        dist_norm = np.ones((H, W), dtype=np.float32)
        valid_d = np.isfinite(dist_grid)
        dist_norm[valid_d] = (dist_grid[valid_d] / diagonal).astype(np.float32)
        
        # Ensure vectors are zero in unreachable areas
        vec_x[~reachable_mask] = 0
        vec_y[~reachable_mask] = 0
        
        # Re-normalize vectors (subsampling checks)
        magnitude = np.sqrt(vec_x**2 + vec_y**2)
        valid_mag = magnitude > 1e-8
        vec_x[valid_mag] /= magnitude[valid_mag]
        vec_y[valid_mag] /= magnitude[valid_mag]
        
        return GeodesicGuidance(
            vec_x=vec_x,
            vec_y=vec_y,
            dist_normalized=dist_norm,
            reachable=reachable_mask.astype(np.float32)
        )

    @staticmethod
    def _empty_result(H: int, W: int) -> GeodesicGuidance:
        """Return an empty result (for unreachable cases)."""
        return GeodesicGuidance(
            vec_x=np.zeros((H, W), dtype=np.float32),
            vec_y=np.zeros((H, W), dtype=np.float32),
            dist_normalized=np.ones((H, W), dtype=np.float32),
            reachable=np.zeros((H, W), dtype=np.float32)
        )
