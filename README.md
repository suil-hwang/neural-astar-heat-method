# Neural A\* with Heat Method: Enhancing Path Planning via Physics-Informed Geometric Guidance

## Overview

A novel path planning framework that enhances Neural A\* with **Heat Method-based Geodesic Supervision** and **Physics-Informed Geometric Losses**.

**Core Ideas:**

- **Heat Method** (Crane et al., 2017): Computes true Geodesic Distance that respects obstacles via heat diffusion, and extracts goal-directed Vector Field from its gradient
- **Multi-Head Encoder**: Separates Cost Map prediction (main task) from Distance/Vector prediction (auxiliary tasks) to avoid gradient conflicts
- **Physics-Informed Losses**: Enforces geometric consistency through Eikonal regularization and gradient-direction alignment
- **Uncertainty Weighting** (Kendall et al., 2018): Automatically balances multiple loss terms without manual tuning

**Differences from Neural A\*:**

| Aspect         | Neural A\*       | Ours                                  |
| -------------- | ---------------- | ------------------------------------- |
| Encoder Output | Single Cost Head | Multi-Head (Cost + Distance + Vector) |
| Supervision    | Path Loss only   | Path Loss + Geodesic Supervision      |
| Loss Design    | L1 Path Loss     | Physics-Informed Geometric Losses     |
| Loss Balancing | Manual weights   | Uncertainty Weighting (auto)          |

### Architecture

#### 1. Heat Method → Geodesic Guidance Generation (Preprocessing)

- Computes **Geodesic Distance Field** by solving heat diffusion from the goal point
- Generates **Vector Guidance Field (Vx, Vy)** by normalizing the negative gradient of distance
- These fields encode **global topological information** representing "shortest path direction to goal"

#### 2. Multi-Head Geodesic Encoder

- **Shared U-Net Backbone**: Learns spatial features from Map + Start + Goal
- **Cost Head**: Produces guidance cost map for A\* search (Main Task)
- **Distance Head**: Predicts geodesic distance for auxiliary supervision
- **Vector Head**: Predicts unit direction vectors toward the goal

**Head Separation Rationale:**

- Cost map represents **relative priority** for A\* exploration
- Distance represents **absolute metric** to the goal
- Single head cannot satisfy both without gradient conflicts

#### 3. Physics-Informed Geometric Losses

| Loss Component       | Formula         | Purpose                             |
| -------------------- | --------------- | ----------------------------------- |
| **Distance Loss**    | Log-space L1    | Scale-invariant distance regression |
| **Vector Loss**      | 1 - cos(θ)      | Direction accuracy over magnitude   |
| **Consistency Loss** | ∇D̂ ≈ -V̂         | Gradient-vector alignment          |
| **Eikonal Loss**     | \|∇D̂\| ≈ 1      | Distance field regularity          |

#### 4. Overall Pipeline

```text
[Preprocessing - Offline]
Map + Goal → Heat Method → Distance Field + Vector Field

[Training]
Input: Map + Start + Goal
         ↓
    Multi-Head U-Net Encoder
         ↓
    Cost Map + Pred_Dist + Pred_Vec
         ↓
    Differentiable A* Search (using Cost Map)
         ↓
    Path Loss + Geometric Loss (auto-weighted)
         ↓
    Backpropagation

[Inference]
Input: Map + Start + Goal
         ↓
    Encoder → Cost Map (Distance/Vector heads ignored)
         ↓
    A* Search → Optimal Path
```

### Training Modes

| Mode           | Description           | Encoder             | Supervision           |
| -------------- | --------------------- | ------------------- | --------------------- |
| `vanilla`      | Standard A\* baseline | -                   | -                     |
| `neural_astar` | Original Neural A\*   | U-Net (Single Head) | Path Loss             |
| `ours`         | Proposed method       | MultiHeadGeoUnet    | Path + Geometric Loss |

---

## Installation

### Create Conda Environment

```bash
conda env create -f environment.yml
conda activate neural-astar
```

### Install Package

```bash
pip install -e .
```

---

## Usage

### 1. Dataset Preprocessing (Heat Method)

Compute Geodesic Distance Field and Vector Field for existing datasets:

```bash
# 32x32 maze
python heat_method/preprocessing.py \
    --data_path data/maze/mazes_032_moore_c8.npz \
    --output_path data/maze_preprocessed/mazes_032_moore_c8_ours.npz

# 64x64 mixed maze
python heat_method/preprocessing.py \
    --data_path data/maze/mixed_064_moore_c16.npz \
    --output_path data/maze_preprocessed/mixed_064_moore_c16_ours.npz
```

**Extended NPZ File Structure:**

| Key Name            | Shape          | Description                                  |
| ------------------- | -------------- | -------------------------------------------- |
| `arr_0`~`arr_3`     | Original       | Train (800): Map, Goal, OptPolicy, OptDist   |
| `arr_4`~`arr_7`     | Original       | Valid (100): Same as above                   |
| `arr_8`~`arr_11`    | Original       | Test (100~400): Same as above                |
| `{split}_vec_x`     | `(N, 1, H, W)` | X-direction unit vector (toward goal)        |
| `{split}_vec_y`     | `(N, 1, H, W)` | Y-direction unit vector (toward goal)        |
| `{split}_dist`      | `(N, 1, H, W)` | Geodesic distance (normalized by diagonal)   |
| `{split}_reachable` | `(N, 1, H, W)` | Reachability mask (1: reachable, 0: blocked) |

> `{split}` = `train`, `valid`, `test`

### 2. Training

```bash
# Neural A* (Original - Baseline)
python scripts/train.py mode=neural_astar encoder.arch=Unet \
    dataset=data/maze_preprocessed/mazes_032_moore_c8_ours

# Ours (Multi-Head with Geodesic Supervision)
python scripts/train.py mode=ours encoder.arch=MultiHeadGeoUnet \
    dataset=data/maze_preprocessed/mazes_032_moore_c8_ours
```

**Key Training Parameters:**

| Parameter                          | Default | Description                |
| ---------------------------------- | ------- | -------------------------- |
| `params.path_loss_weight`          | 100     | Weight for path loss       |
| `params.geo_loss_weight`           | 0.1     | Weight for geometric loss  |
| `params.use_uncertainty_weighting` | false   | Enable auto loss balancing |
| `params.use_weno_gradient`         | true    | Use WENO3 for Eikonal loss |

### 3. Model Comparison

Compare trained models on the test set:

```bash
# 32x32 dataset
python scripts/compare_models.py \
    --model-dir model \
    --dataset data/maze_preprocessed/mazes_032_moore_c8_ours

# 64x64 dataset
python scripts/compare_models.py \
    --model-dir model \
    --dataset data/maze_preprocessed/mixed_064_moore_c16_ours
```

**Evaluation Metrics:**

| Metric        | Description                                                   |
| ------------- | ------------------------------------------------------------- |
| **Opt (%)**   | Ratio of finding optimal-length paths (higher is better)      |
| **Exp (%)**   | Node exploration reduction vs. Vanilla A\* (higher is better) |
| **Hmean**     | Harmonic mean of Opt and Exp (higher is better)               |
| **Time (ms)** | Inference time per sample (lower is better)                   |

### 4. Visualization (GIF Generation)

Generate comparison GIFs showing search process:

```bash
python scripts/create_comparison_gif.py \
    dataset=data/maze_preprocessed/mazes_032_moore_c8_ours \
    problem_id=0 \
    resultdir=results
```

**Output Files:**

```text
results/comparison/
├── vanilla_{dataset}_{id}.gif         # A* only
├── neural_astar_{dataset}_{id}.gif    # Neural A* only
├── ours_{dataset}_{id}.gif            # Ours only
└── comparison_{dataset}_{id}.gif      # Side-by-side comparison
```

---

## Experimental Results

> **Note:** All experiments were trained for **50 epochs**.

### 32×32 Maze Dataset (mazes_032_moore_c8, 100 Test Samples)

| Model       | Opt (%) ↑ | Exp (%) ↑ | Hmean ↑   | Time (ms) ↓ |
| ----------- | --------- | --------- | --------- | ----------- |
| Vanilla A\* | 100.00    | 0.00      | 0.00      | 13.35       |
| Neural A\*  | 55.00     | 26.46     | 35.73     | 12.49       |
| **Ours**    | **78.00** | **35.46** | **48.75** | **8.88**    |

### 64×64 Mixed Maze Dataset (mixed_064_moore_c16, 400 Test Samples)

| Model       | Opt (%) ↑ | Exp (%) ↑ | Hmean ↑   | Time (ms) ↓ |
| ----------- | --------- | --------- | --------- | ----------- |
| Vanilla A\* | 100.00    | 0.00      | 0.00      | 37.99       |
| Neural A\*  | 65.25     | 27.83     | 39.02     | 23.24       |
| **Ours**    | **77.00** | **27.57** | **40.60** | **29.22**   |

### 64×64 All Maze Dataset (all_064_moore_c16, 400 Test Samples)

| Model       | Opt (%) ↑ | Exp (%) ↑ | Hmean ↑   | Time (ms) ↓ |
| ----------- | --------- | --------- | --------- | ----------- |
| Vanilla A\* | 100.00    | 0.00      | 0.00      | 53.13       |
| Neural A\*  | 47.50     | 41.53     | 44.31     | 36.20       |
| **Ours**    | **70.00** | **36.75** | **48.19** | **48.22**   |

---

## References

1. Yonetani, R., Taniai, T., et al. "Path Planning using Neural A*Search", _ICML_, 2021.
   [[paper]](https://arxiv.org/abs/2009.07476) [[project]](https://omron-sinicx.github.io/neural-astar/)

2. Crane, K., Weischedel, C., Wardetzky, M. "The Heat Method for Distance Computation", _ACM ToG_, 2017.
   [[paper]](https://dl.acm.org/doi/10.1145/3131280) [[project]](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)

3. Kendall, A., Gal, Y., Cipolla, R. "Multi-Task Learning Using Uncertainty to Weigh Losses", _CVPR_, 2018.
   [[paper]](https://arxiv.org/abs/1705.07115)

4. Jiang, G.-S., Peng, D. "Weighted ENO Schemes for Hamilton-Jacobi Equations", _SIAM J. Sci. Comput._, 2000.
   [[paper]](https://doi.org/10.1137/S106482759732455X)
