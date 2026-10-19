"""Probe linear decodability along S -> T for the radius-18 Ring task.

Run from bottleneck/:
    python ring_linear_probe.py --radius 18 --model-type SW
    python ring_linear_probe.py --radius 18 --model-type SW --checkpoint /path/to/model.ckpt

The generator has only five class prototypes. Accuracy measures in-sample
decodability, NOT generalization to unseen graphs.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from data_generate.graphs_generation import RingDataset
from models.graph_model import GraphModel
from utils import fix_seed, get_args


def ring_paths(radius):
    """Two shortest paths from node 0 (S) to node radius (T)."""
    if radius < 1:
        raise ValueError("radius must be positive")
    return {
        "clockwise": list(range(radius + 1)),
        "counterclockwise": [(-d) % (2 * radius) for d in range(radius + 1)],
    }


class LinearProbeBank(nn.Module):
    """Independent single-layer affine linear classifiers for each path node."""

    def __init__(self, positions, hidden_dim, classes):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, classes) for _ in range(positions)]
        )

    def forward(self, features):
        # [examples, positions, hidden_dim] -> [examples, positions, classes]
        return torch.stack(
            [head(features[:, i]) for i, head in enumerate(self.heads)], dim=1
        )


def make_ring_batch(args, device):
    train, _, _ = RingDataset(args=args, add_crosses=False).generate_data()
    if len(train) != 5 or sorted(int(g.y) for g in train) != list(range(5)):
        raise ValueError("Expected five distinct class prototypes in the original Ring task")
    batch = Batch.from_data_list(train).to(device)
    batch.root_mask = batch.train_mask
    return batch, len(train)


def read_checkpoint(path):
    """Accept Lightning checkpoint or GraphModel weights saved by this script."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload.get("model_state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError(f"Unrecognized checkpoint: {path}")
    if all(key.startswith("model.") for key in state):
        state = {key[len("model."):]: val for key, val in state.items()}
    return state


def train_gnn(model, batch, args, epochs):
    """Train the same supervised Ring objective as train.py."""
    optimizer = getattr(torch.optim, args.optim_type)(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor
    )
    labels = batch.y.long()
    best = (-1.0, float("-inf"))
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch), labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            accuracy = (logits.argmax(-1) == labels).float().mean().item()
            eval_loss = F.cross_entropy(logits, labels).item()
        scheduler.step(accuracy)
        if (accuracy, -eval_loss) > best:
            best = (accuracy, -eval_loss)
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"GNN epoch {epoch:4d}/{epochs} | loss {eval_loss:.5f} | "
                  f"target accuracy {accuracy:.1%}", flush=True)
    model.load_state_dict(best_state)
    return best[0]


@torch.no_grad()
def extract_embeddings(model, batch, examples, radius):
    model.eval()
    all_nodes = model.compute_node_embedding(batch)
    if all_nodes.shape[0] != examples * 2 * radius:
        raise ValueError("Unexpected number of node embeddings")
    return all_nodes.reshape(examples, 2 * radius, -1).detach(), batch.y.long()


def fit_probes(features, labels, classes, epochs=2000, lr=0.05):
    """Fit per-position linear heads; report accuracy, margin and class distances.

    Per-position centering and scalar scaling are affine, hence equivalent to
    training an affine linear head on the original (unscaled) embeddings.
    They also make small but nonzero embeddings easier to linearly decode.
    """
    examples, positions, width = features.shape
    centered = features - features.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=(0, 2), keepdim=True).sqrt()
    features_scaled = centered / scale.clamp_min(torch.finfo(features.dtype).tiny)
    probes = LinearProbeBank(positions, width, classes).to(features.device)
    optimizer = torch.optim.Adam(probes.parameters(), lr=lr)
    expected = labels[:, None].expand(examples, positions)
    perfect_streak = 0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = probes(features_scaled)
        loss = F.cross_entropy(logits.reshape(-1, classes), expected.reshape(-1))
        loss.backward()
        optimizer.step()
        if (logits.argmax(-1) == expected).all().item():
            perfect_streak += 1
            if perfect_streak >= 30:
                break
        else:
            perfect_streak = 0

    with torch.no_grad():
        logits = probes(features_scaled)
        accuracy = (logits.argmax(-1) == expected).float().mean(dim=0)
        correct = logits.gather(-1, expected[..., None]).squeeze(-1)
        incorrect = logits.masked_fill(
            F.one_hot(expected, classes).bool(), float("-inf")
        ).max(dim=-1).values
        margins = (correct - incorrect).min(dim=0).values
        min_distances, relative_distances = [], []
        for position in range(positions):
            points = features[:, position, :]
            minimum = torch.pdist(points, p=2).min().item()
            average_norm = points.norm(dim=-1).mean().item()
            min_distances.append(minimum)
            relative_distances.append(minimum / max(average_norm, 1e-30))
    return (accuracy.cpu().tolist(), margins.cpu().tolist(),
            min_distances, relative_distances)


def save_plots(rows, directory):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: results.csv saved, plots skipped")
        return
    for metric, ylabel, filename in (
        ("probe_accuracy", "Linear probe accuracy", "accuracy.png"),
        ("min_pairwise_distance", "Minimum pairwise class distance", "distance.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for path, iterator in itertools.groupby(rows, key=lambda item: item["path"]):
            values = list(iterator)
            ax.plot([item["distance_from_s"] for item in values],
                    [item[metric] for item in values], marker="o", label=path)
        if metric == "probe_accuracy":
            ax.axhline(0.2, linestyle="--", label="chance (5 classes)")
            ax.set_ylim(0, 1.05)
        ax.set_xlabel("Distance from S (T = radius)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=18)
    parser.add_argument("--model-type", default="SW",
                        choices=["SW", "GIN", "GCN", "GAT", "SAGE", "Transformer"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gnn-epochs", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=2000)
    parser.add_argument("--probe-lr", type=float, default=0.05)
    parser.add_argument("--failure-threshold", type=float, default=1.0,
                        help="Probe accuracy below this threshold counts as failure")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    cli = parser.parse_args()
    if not (0.0 < cli.failure_threshold <= 1.0):
        parser.error("--failure-threshold must be in (0, 1]")
    if cli.radius < 1 or cli.probe_epochs < 1 or (
        cli.gnn_epochs is not None and cli.gnn_epochs < 1
    ):
        parser.error("radius and training epochs must be positive")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cli.device == "auto" else cli.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    args, _ = get_args(cli.radius, cli.model_type, "Ring")
    args.seed = cli.seed
    fix_seed(cli.seed)
    batch, examples = make_ring_batch(args, device)
    model = GraphModel(args).to(device)
    output = cli.output_dir or (
        Path("data") / "probes" / "Ring" / cli.model_type /
        f"Radius_{cli.radius}" / f"seed_{cli.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if cli.checkpoint:
        model.load_state_dict(read_checkpoint(cli.checkpoint), strict=True)
        print(f"Loaded trained GNN checkpoint: {cli.checkpoint}")
    else:
        train_gnn(model, batch, args, cli.gnn_epochs or args.max_epochs)
        torch.save(
            {"model_state_dict": model.state_dict(),
             "radius": cli.radius, "model_type": cli.model_type},
            output / "gnn.pt",
        )
        print(f"Saved frozen GNN weights: {output / 'gnn.pt'}")

    model.eval()
    with torch.no_grad():
        target_accuracy = (
            model(batch).argmax(dim=-1) == batch.y
        ).float().mean().item()
    print(f"Original frozen GNN at T: {target_accuracy:.1%} accuracy")
    embeddings, labels = extract_embeddings(model, batch, examples, cli.radius)

    rows = []
    failure_summary = {}
    for path, nodes in ring_paths(cli.radius).items():
        accuracy, margins, min_distances, relative = fit_probes(
            embeddings[:, nodes, :].clone(), labels, args.out_dim,
            epochs=cli.probe_epochs, lr=cli.probe_lr
        )
        failed = [d for d, value in enumerate(accuracy)
                  if value + 1e-6 < cli.failure_threshold]
        first = failed[0] if failed else None
        failure_summary[path] = {
            "first_failed_distance": first,
            "first_failed_node": (nodes[first] if first is not None else None),
            "failed_distances": failed,
            "probe_accuracy_at_T": accuracy[-1],
        }
        for distance, node_id in enumerate(nodes):
            rows.append({
                "path": path, "node_id": node_id,
                "distance_from_s": distance, "probe_accuracy": accuracy[distance],
                "minimum_margin": margins[distance],
                "min_pairwise_distance": min_distances[distance],
                "relative_min_pairwise_distance": relative[distance],
                "probe_failed": accuracy[distance] + 1e-6 < cli.failure_threshold,
            })
            print(f"{path:17s} d={distance:2d} node={node_id:2d} "
                  f"probe accuracy={accuracy[distance]:.1%} "
                  f"min class distance={min_distances[distance]:.3g}")

    with (output / "probes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "task": "Ring", "radius": cli.radius, "gnn_type": cli.model_type,
        "source_node": 0, "target_node": cli.radius,
        "original_GNN_accuracy_at_T": target_accuracy,
        "failure_threshold": cli.failure_threshold,
        "paths": failure_summary,
    }
    with (output / "failure_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    save_plots(rows, output)
    for path, result in failure_summary.items():
        if result["first_failed_distance"] is None:
            print(f"{path}: no linear-probe failures along S -> T "
                  f"(all positions reach {cli.failure_threshold:.0%})")
        else:
            print(f"{path}: FIRST PROBE FAILURE at distance "
                  f"{result['first_failed_distance']}, "
                  f"node {result['first_failed_node']}; "
                  f"all failed distances: {result['failed_distances']}")
    print(f"Saved results to {output.resolve()}")
    print("NOTE: Probe failure means failed linear decoding with this optimizer; it "
          "does not prove that the information has vanished. This is in-sample "
          "decodability of five fixed graphs, not generalization.")


if __name__ == "__main__":
    main()
