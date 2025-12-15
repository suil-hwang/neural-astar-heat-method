# heat_method/validation.py
from __future__ import annotations

from typing import Tuple, Literal

import numpy as np
from numpy.typing import NDArray



# Constants
_MIN_VECTOR_MAG: float = 1e-6 # Minimum vector magnitude to consider as valid direction
_DEFAULT_MAX_STEPS: int = 2000
_DEFAULT_STEP_SIZE: float = 0.5
_DEFAULT_TOLERANCE: float = 1.5
_DEFAULT_SAMPLE_LIMIT: int = 100


def _sample_vector_bilinear(
    vec_x: NDArray[np.float32],
    vec_y: NDArray[np.float32],
    r: float,
    c: float,
) -> Tuple[float, float]:
    """Sample vector field with bilinear interpolation."""
    H, W = vec_x.shape
    
    # Clamp to valid range
    r = np.clip(r, 0, H - 1 - 1e-6)
    c = np.clip(c, 0, W - 1 - 1e-6)
    
    # Get integer and fractional parts
    r0 = int(r)
    c0 = int(c)
    r1 = min(r0 + 1, H - 1)
    c1 = min(c0 + 1, W - 1)
    
    fr = r - r0
    fc = c - c0
    
    # Bilinear interpolation weights
    w00 = (1 - fr) * (1 - fc)
    w01 = (1 - fr) * fc
    w10 = fr * (1 - fc)
    w11 = fr * fc
    
    vx = w00 * vec_x[r0, c0] + w01 * vec_x[r0, c1] + \
         w10 * vec_x[r1, c0] + w11 * vec_x[r1, c1]
    vy = w00 * vec_y[r0, c0] + w01 * vec_y[r0, c1] + \
         w10 * vec_y[r1, c0] + w11 * vec_y[r1, c1]
    
    return float(vx), float(vy)


def _sample_vector_nearest(
    vec_x: NDArray[np.float32],
    vec_y: NDArray[np.float32],
    r: float,
    c: float,
) -> Tuple[float, float]:
    """Sample vector field with nearest neighbor interpolation."""
    H, W = vec_x.shape
    ir = int(round(r))
    ic = int(round(c))
    
    # Clamp to bounds
    ir = max(0, min(ir, H - 1))
    ic = max(0, min(ic, W - 1))
    
    return float(vec_x[ir, ic]), float(vec_y[ir, ic])


def _trace_path_euler(
    vec_x: NDArray[np.float32],
    vec_y: NDArray[np.float32],
    start_r: float,
    start_c: float,
    goal_r: int,
    goal_c: int,
    max_steps: int,
    step_size: float,
    tolerance: float,
    use_bilinear: bool = False,
) -> bool:
    """Trace path using Euler integration."""
    H, W = vec_x.shape
    curr_r, curr_c = start_r, start_c
    tolerance_sq = tolerance * tolerance
    
    for _ in range(max_steps):
        # Check goal reached
        dr = curr_r - goal_r
        dc = curr_c - goal_c
        if dr * dr + dc * dc < tolerance_sq:
            return True
        
        # Sample vector at current position
        if use_bilinear:
            vx, vy = _sample_vector_bilinear(vec_x, vec_y, curr_r, curr_c)
        else:
            ir = int(round(curr_r))
            ic = int(round(curr_c))
            
            # Bounds check
            if not (0 <= ir < H and 0 <= ic < W):
                return False
            
            vx = float(vec_x[ir, ic])
            vy = float(vec_y[ir, ic])
        
        # Check for zero vector (stuck)
        if abs(vx) < _MIN_VECTOR_MAG and abs(vy) < _MIN_VECTOR_MAG:
            return False
        
        # Euler step: position += velocity * dt
        curr_c += vx * step_size
        curr_r += vy * step_size
        
        # Bounds check after step
        if not (0 <= curr_r < H and 0 <= curr_c < W):
            return False
    
    return False


def _trace_path_rk4(
    vec_x: NDArray[np.float32],
    vec_y: NDArray[np.float32],
    start_r: float,
    start_c: float,
    goal_r: int,
    goal_c: int,
    max_steps: int,
    step_size: float,
    tolerance: float,
) -> bool:
    """Trace path using 4th-order Runge-Kutta integration."""
    H, W = vec_x.shape
    curr_r, curr_c = start_r, start_c
    tolerance_sq = tolerance * tolerance
    h = step_size
    
    for _ in range(max_steps):
        # Check goal reached
        dr = curr_r - goal_r
        dc = curr_c - goal_c
        if dr * dr + dc * dc < tolerance_sq:
            return True
        
        # Bounds check
        if not (0 <= curr_r < H - 1e-6 and 0 <= curr_c < W - 1e-6):
            return False
        
        # k1: velocity at current position
        k1_x, k1_y = _sample_vector_bilinear(vec_x, vec_y, curr_r, curr_c)
        
        if abs(k1_x) < _MIN_VECTOR_MAG and abs(k1_y) < _MIN_VECTOR_MAG:
            return False
        
        # k2: velocity at half-step using k1
        r2 = curr_r + 0.5 * h * k1_y
        c2 = curr_c + 0.5 * h * k1_x
        if not (0 <= r2 < H and 0 <= c2 < W):
            # Fall back to Euler step
            curr_r += h * k1_y
            curr_c += h * k1_x
            continue
        k2_x, k2_y = _sample_vector_bilinear(vec_x, vec_y, r2, c2)
        
        # k3: velocity at half-step using k2
        r3 = curr_r + 0.5 * h * k2_y
        c3 = curr_c + 0.5 * h * k2_x
        if not (0 <= r3 < H and 0 <= c3 < W):
            curr_r += h * k1_y
            curr_c += h * k1_x
            continue
        k3_x, k3_y = _sample_vector_bilinear(vec_x, vec_y, r3, c3)
        
        # k4: velocity at full step using k3
        r4 = curr_r + h * k3_y
        c4 = curr_c + h * k3_x
        if not (0 <= r4 < H and 0 <= c4 < W):
            curr_r += h * k1_y
            curr_c += h * k1_x
            continue
        k4_x, k4_y = _sample_vector_bilinear(vec_x, vec_y, r4, c4)
        
        # RK4 weighted average
        curr_c += (h / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        curr_r += (h / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    
    return False


def check_reachability(
    vec_x: NDArray[np.float32],
    vec_y: NDArray[np.float32],
    start_map: NDArray,
    goal_pos: Tuple[int, int],
    max_steps: int = _DEFAULT_MAX_STEPS,
    step_size: float = _DEFAULT_STEP_SIZE,
    tolerance: float = _DEFAULT_TOLERANCE,
    sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
    method: Literal["euler", "rk4"] = "rk4",
) -> float:
    """Verify vector field guides to goal via greedy path tracing."""
    g_r, g_c = goal_pos
    
    # Get valid start positions
    starts = np.argwhere(start_map == 1)
    if len(starts) == 0:
        return 0.0
    
    # Random sampling if too many starts
    if len(starts) > sample_limit:
        rng = np.random.default_rng()
        indices = rng.choice(len(starts), sample_limit, replace=False)
        starts = starts[indices]
    
    # Count successful paths
    success_count = 0
    tolerance_sq = tolerance * tolerance
    
    for r_start, c_start in starts:
        # Skip if already at goal
        dr = r_start - g_r
        dc = c_start - g_c
        if dr * dr + dc * dc < tolerance_sq:
            success_count += 1
            continue
        
        # Trace path using selected method
        if method == "rk4":
            reached = _trace_path_rk4(
                vec_x, vec_y,
                float(r_start), float(c_start),
                g_r, g_c,
                max_steps, step_size, tolerance
            )
        else:
            reached = _trace_path_euler(
                vec_x, vec_y,
                float(r_start), float(c_start),
                g_r, g_c,
                max_steps, step_size, tolerance,
                use_bilinear=True
            )
        
        if reached:
            success_count += 1
    
    return success_count / len(starts) if len(starts) > 0 else 0.0


def trace_path(
    vec_x: NDArray[np.float32],
    vec_y: NDArray[np.float32],
    start_pos: Tuple[float, float],
    goal_pos: Tuple[int, int],
    max_steps: int = _DEFAULT_MAX_STEPS,
    step_size: float = _DEFAULT_STEP_SIZE,
    tolerance: float = _DEFAULT_TOLERANCE,
    method: Literal["euler", "rk4"] = "rk4",
) -> Tuple[NDArray[np.float64], bool]:
    """Trace a single path through the vector field."""
    H, W = vec_x.shape
    curr_r, curr_c = float(start_pos[0]), float(start_pos[1])
    g_r, g_c = goal_pos
    tolerance_sq = tolerance * tolerance
    h = step_size
    
    path = [(curr_r, curr_c)]
    reached = False
    
    for _ in range(max_steps):
        # Check goal reached
        dr = curr_r - g_r
        dc = curr_c - g_c
        if dr * dr + dc * dc < tolerance_sq:
            reached = True
            break
        
        # Bounds check
        if not (0 <= curr_r < H - 1e-6 and 0 <= curr_c < W - 1e-6):
            break
        
        if method == "rk4":
            # RK4 integration step
            k1_x, k1_y = _sample_vector_bilinear(vec_x, vec_y, curr_r, curr_c)
            
            if abs(k1_x) < _MIN_VECTOR_MAG and abs(k1_y) < _MIN_VECTOR_MAG:
                break
            
            # k2
            r2, c2 = curr_r + 0.5*h*k1_y, curr_c + 0.5*h*k1_x
            if 0 <= r2 < H and 0 <= c2 < W:
                k2_x, k2_y = _sample_vector_bilinear(vec_x, vec_y, r2, c2)
            else:
                k2_x, k2_y = k1_x, k1_y
            
            # k3
            r3, c3 = curr_r + 0.5*h*k2_y, curr_c + 0.5*h*k2_x
            if 0 <= r3 < H and 0 <= c3 < W:
                k3_x, k3_y = _sample_vector_bilinear(vec_x, vec_y, r3, c3)
            else:
                k3_x, k3_y = k2_x, k2_y
            
            # k4
            r4, c4 = curr_r + h*k3_y, curr_c + h*k3_x
            if 0 <= r4 < H and 0 <= c4 < W:
                k4_x, k4_y = _sample_vector_bilinear(vec_x, vec_y, r4, c4)
            else:
                k4_x, k4_y = k3_x, k3_y
            
            # Weighted update
            curr_c += (h / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            curr_r += (h / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        else:
            # Euler integration step
            vx, vy = _sample_vector_bilinear(vec_x, vec_y, curr_r, curr_c)
            
            if abs(vx) < _MIN_VECTOR_MAG and abs(vy) < _MIN_VECTOR_MAG:
                break
            
            curr_c += h * vx
            curr_r += h * vy
        
        path.append((curr_r, curr_c))
    
    return np.array(path, dtype=np.float64), reached