# scripts/visualize_cost_map.py
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import load_from_ptl_checkpoint


def visualize_cost_maps(
    dataset_path: str,
    model_path: str,
    sample_indices: list = (0, 1, 2),
    save_dir: str = "cost_map_vis",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Visualize cost maps (Geo-Attention, 3-channel input)"""
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_path}.npz")
    encoder_input = "msg"  # 3 channels (map, start, goal)

    test_loader = create_dataloader(
        dataset_path + ".npz",
        "test",
        batch_size=1,
        shuffle=False,
        geo_supervision=True,
    )

    data = np.load(dataset_path + ".npz")
    test_dist = data["test_dist"]        # [N, 1, H, W]
    test_reachable = data["test_reachable"]  # [N, 1, H, W]

    print(f"Loading model from: {model_path}")
    neural_astar = NeuralAstar(
        encoder_input=encoder_input,
        encoder_arch="GeoAttentionUnet",
        encoder_depth=4,
        learn_obstacles=False,
        Tmax=1.0,
    )

    state_dict = load_from_ptl_checkpoint(model_path)
    neural_astar.load_state_dict(state_dict)
    neural_astar.to(device)
    neural_astar.eval()

    print(f"\nVisualizing samples: {sample_indices}")

    all_batches = list(test_loader)
    for idx in sample_indices:
        if idx >= len(all_batches):
            print(f"Sample {idx} not available (max: {len(all_batches) - 1})")
            continue

        batch = all_batches[idx]
        map_design, start_map, goal_map, opt_traj, dist, reachable = batch

        map_design = map_design.to(device)
        start_map = start_map.to(device)
        goal_map = goal_map.to(device)
        dist = dist.to(device)
        reachable = reachable.to(device)

        with torch.no_grad():
            cost_map = neural_astar.encode(map_design, start_map, goal_map)
            pred_vec_field = neural_astar.get_vector_field()
            outputs = neural_astar.astar(cost_map, start_map, goal_map, map_design)

        map_np = map_design[0, 0].cpu().numpy()
        start_np = start_map[0, 0].cpu().numpy()
        goal_np = goal_map[0, 0].cpu().numpy()
        cost_np = cost_map[0, 0].cpu().numpy()
        dist_np = test_dist[idx, 0]
        reachable_np = test_reachable[idx, 0]
        opt_traj_np = opt_traj[0, 0].cpu().numpy()
        pred_path_np = outputs.paths[0, 0].cpu().numpy()
        pred_hist_np = outputs.histories[0, 0].cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Sample {idx} | Geo-Attention (3ch)", fontsize=14)

        ax = axes[0, 0]
        ax.imshow(map_np, cmap='gray')
        start_pos = np.argwhere(start_np > 0.5)
        goal_pos = np.argwhere(goal_np > 0.5)
        if len(start_pos) > 0:
            ax.scatter(start_pos[0, 1], start_pos[0, 0], c='blue', s=100, marker='o', label='Start')
        if len(goal_pos) > 0:
            ax.scatter(goal_pos[0, 1], goal_pos[0, 0], c='red', s=100, marker='*', label='Goal')
        ax.set_title("Map + Start/Goal")
        ax.legend()
        ax.axis('off')

        ax = axes[0, 1]
        dist_masked = np.ma.masked_where(reachable_np < 0.5, dist_np)
        im = ax.imshow(dist_masked, cmap='viridis')
        ax.imshow(map_np == 0, cmap='gray', alpha=0.3)
        ax.set_title(
            f"Geodesic Distance\nmin={dist_np[reachable_np>0.5].min():.2f}, "
            f"max={dist_np[reachable_np>0.5].max():.2f}"
        )
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        ax = axes[0, 2]
        ax.imshow(reachable_np, cmap='RdYlGn')
        ax.set_title("Reachable Mask")
        ax.axis('off')

        ax = axes[0, 3]
        if pred_vec_field is not None and pred_vec_field[0] is not None:
            vx = pred_vec_field[0][0, 0].cpu().numpy()
            vy = pred_vec_field[1][0, 0].cpu().numpy()
            stride = max(1, map_np.shape[0] // 16)
            X, Y = np.meshgrid(
                np.arange(0, map_np.shape[1], stride),
                np.arange(0, map_np.shape[0], stride),
            )
            U = vx[0::stride, 0::stride]
            V = vy[0::stride, 0::stride]
            ax.imshow(map_np, cmap="gray")
            ax.quiver(X, Y, U, -V, color="red", scale=20, headwidth=3)
            ax.set_title("Predicted Vector Field")
            ax.axis("off")
        else:
            ax.text(
                0.5,
                0.5,
                "Vector Field\nN/A",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_title("Vector Field (N/A)")
            ax.axis("off")

        ax = axes[1, 0]
        cost_masked = np.ma.masked_where(map_np < 0.5, cost_np)
        im = ax.imshow(cost_masked, cmap='hot')
        ax.set_title(
            f"Cost Map (Model Output)\n"
            f"min={cost_np[map_np>0.5].min():.3f}, max={cost_np[map_np>0.5].max():.3f}"
        )
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        ax = axes[1, 1]
        ax.imshow(cost_masked, cmap='hot')
        opt_path_overlay = np.ma.masked_where(opt_traj_np < 0.5, opt_traj_np)
        ax.imshow(opt_path_overlay, cmap='Blues', alpha=0.8)
        ax.set_title("Cost Map + Optimal Path (Blue)")
        ax.axis('off')

        ax = axes[1, 2]
        ax.imshow(map_np, cmap='gray')
        hist_overlay = np.ma.masked_where(pred_hist_np < 0.5, pred_hist_np)
        ax.imshow(hist_overlay, cmap='Greens', alpha=0.7)
        exp_count = pred_hist_np.sum()
        ax.set_title(f"Exploration History\n({int(exp_count)} nodes)")
        ax.axis('off')

        ax = axes[1, 3]
        ax.imshow(map_np, cmap='gray')
        opt_overlay = np.ma.masked_where(opt_traj_np < 0.5, opt_traj_np)
        ax.imshow(opt_overlay, cmap='Blues', alpha=0.6)
        pred_overlay = np.ma.masked_where(pred_path_np < 0.5, pred_path_np)
        ax.imshow(pred_overlay, cmap='Reds', alpha=0.6)
        opt_len = int(opt_traj_np.sum())
        pred_len = int(pred_path_np.sum())
        is_optimal = "✓ Optimal" if pred_len <= opt_len + 1 else "✗ Suboptimal"
        ax.set_title(f"Paths: Optimal(Blue)={opt_len}, Pred(Red)={pred_len}\n{is_optimal}")
        ax.axis('off')

        plt.tight_layout()
        save_path = f"{save_dir}/cost_map_geo_sample{idx:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    print(f"\nDone! Visualizations saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize Cost Maps (Geo-Attention, 3ch)")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset (without .npz)")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--samples", type=str, default="0,1,2,3,4",
                       help="Comma-separated sample indices")
    parser.add_argument("--save-dir", type=str, default="cost_map_vis",
                       help="Directory to save visualizations")

    args = parser.parse_args()
    sample_indices = [int(x) for x in args.samples.split(",")]

    visualize_cost_maps(
        dataset_path=args.dataset,
        model_path=args.model,
        sample_indices=sample_indices,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()


# 사용법:
# python scripts/visualize_cost_map.py --dataset data/maze_preprocessed/mazes_032_moore_c8_ours --model model/ours/mazes_032_moore_c8_ours --samples 0,1,2,3,4

