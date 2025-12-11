# heat_method/validation.py
from __future__ import annotations

import numpy as np
from typing import Tuple


def check_reachability(
    vec_x: np.ndarray,
    vec_y: np.ndarray,
    start_map: np.ndarray,
    goal_pos: Tuple[int, int],
    max_steps: int = 2000,
    step_size: float = 0.5,
    tolerance: float = 1.5,
    sample_limit: int = 100,
) -> float:
    """
    Verify vector field guides to goal via greedy path tracing.
    
    Simulates movement from random start points following the vector field.
    Returns success rate (ratio of paths reaching the goal).
    """
    rows, cols = vec_x.shape
    g_r, g_c = goal_pos
    
    starts = np.argwhere(start_map == 1)
    if len(starts) == 0:
        return 0.0
        
    if len(starts) > sample_limit:
        indices = np.random.choice(len(starts), sample_limit, replace=False)
        starts = starts[indices]
    
    success_count = 0
    total_tested = len(starts)
    
    for r_start, c_start in starts:
        # Skip if already at goal
        if (r_start - g_r)**2 + (c_start - g_c)**2 < tolerance**2:
            success_count += 1
            continue
            
        curr_r, curr_c = float(r_start), float(c_start)
        reached = False
        
        for _ in range(max_steps):
            ir = int(round(curr_r))
            ic = int(round(curr_c))
            
            # Bounds check
            if not (0 <= ir < rows and 0 <= ic < cols):
                break
                
            # Goal check
            if (curr_r - g_r)**2 + (curr_c - g_c)**2 < tolerance**2:
                reached = True
                break
                
            vx = vec_x[ir, ic]
            vy = vec_y[ir, ic]
            
            # Zero vector means stuck
            if abs(vx) < 1e-9 and abs(vy) < 1e-9:
                break
            
            # Move along vector field
            curr_c += vx * step_size
            curr_r += vy * step_size
            
        if reached:
            success_count += 1
            
    return success_count / total_tested if total_tested > 0 else 0.0
