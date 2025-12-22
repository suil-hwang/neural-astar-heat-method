# Neural A\* with Heat Method: Geometric Guidance for Path Planning

> **AIC6051 Prediction and Planning in Autonomous Driving**  
> 2025 Fall, Hanyang University

A path planning framework that enhances Neural A\* with **Heat Method-based Geodesic Supervision** and **Geometric Losses**.

---

## Key Idea

| Aspect      | Neural A\*         | Ours                                  |
| ----------- | ------------------ | ------------------------------------- |
| Encoder     | Single Cost Head   | Multi-Head (Cost + Distance + Vector) |
| Supervision | Path Loss only     | Path Loss + Geometric Loss            |
| Guidance    | Learned implicitly | Geodesic Distance + Vector Field      |

**Core Components:**

- **Heat Method**: Computes Geodesic Distance and goal-directed Vector Field via heat diffusion
- **Multi-Head Encoder**: Separates Cost prediction (main) from Distance/Vector prediction (auxiliary)
- **Geometric Losses**: Distance Loss, Vector Loss, Gradient-Vector Consistency

---

## Pipeline

```text
[Preprocessing]  Map + Goal  →  Heat Method  →  Distance Field + Vector Field

[Training]       Map + Start + Goal
                        ↓
                 Multi-Head U-Net Encoder
                        ↓
                 Cost Map + Pred_Dist + Pred_Vec
                        ↓
                 Differentiable A* (using Cost Map)
                        ↓
                 Path Loss + Geometric Loss → Backprop

[Inference]      Encoder → Cost Map only → A* Search → Path
```

---

## Results

All experiments trained for **100 epochs** on RTX 4070 Ti Super.

| Dataset     | Model      | Opt(%)↑   | Exp(%)↑   | Hmean↑    |
| ----------- | ---------- | --------- | --------- | --------- |
| 32×32 Maze  | Neural A\* | 74.00     | 26.05     | 38.54     |
|             | **Ours**   | **88.00** | **43.08** | **57.84** |
| 64×64 Mixed | Neural A\* | 65.25     | 27.83     | 39.02     |
|             | **Ours**   | **77.00** | **27.57** | **40.60** |
| 64×64 All   | Neural A\* | 47.50     | 41.53     | 44.31     |
|             | **Ours**   | **70.00** | **36.75** | **48.19** |

**Improvements:** Path Optimality **+12~22%p**, Hmean up to **+50%**

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

### 1. Preprocessing (Heat Method)

```bash
python heat_method/preprocessing.py \
    --data_path data/maze/mazes_032_moore_c8.npz \
    --output_path data/maze_preprocessed/mazes_032_moore_c8_ours.npz
```

### 2. Training

```bash
# Neural A* (Baseline)
python scripts/train.py mode=neural_astar encoder.arch=Unet \
    dataset=data/maze_preprocessed/mazes_032_moore_c8_ours

# Ours
python scripts/train.py mode=ours encoder.arch=MultiHeadGeoUnet \
    dataset=data/maze_preprocessed/mazes_032_moore_c8_ours
```

### 3. Evaluation

```bash
python scripts/compare_models.py \
    --model-dir model \
    --dataset data/maze_preprocessed/mazes_032_moore_c8_ours
```

### 4. Visualization

```bash
python scripts/create_comparison_gif.py \
    dataset=data/maze_preprocessed/mazes_032_moore_c8_ours \
    problem_id=0 resultdir=results
```

---

## References

1. Yonetani et al., "Path Planning using Neural A\*Search", _ICML_, 2021. [[paper]](https://arxiv.org/abs/2009.07476)
2. Crane et al., "The Heat Method for Distance Computation", _ACM ToG_, 2013. [[paper]](https://dl.acm.org/doi/10.1145/3131280)
3. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", _CVPR_, 2018. [[paper]](https://arxiv.org/abs/1705.07115)
