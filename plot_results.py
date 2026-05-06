"""Visualization utility for federated learning experiment results.

Usage:
    # Default: plot metrics.json in current directory (single experiment)
    python plot_results.py

    # Plot specific file(s)
    python plot_results.py metrics_fedprox_cifar10.json
    python plot_results.py metrics_fedavg.json metrics_fedprox.json metrics_fedadagrad.json

    # Plot all metrics*.json in current directory
    python plot_results.py --all

    # Plot all metrics*.json in a specific directory
    python plot_results.py --all --dir results

    # Custom output directory for plots
    python plot_results.py --all --output plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_single_file(filepath: Path) -> tuple:
    """Load a single metrics JSON file. Returns (name, data)."""
    data = json.loads(filepath.read_text())
    if not data:
        return None, None

    # Derive display name from filename
    name = filepath.stem
    if name == "metrics":
        # For plain metrics.json, build name from config
        cfg = data.get("config", {})
        name = f"{cfg.get('strategy', 'exp')}_{cfg.get('dataset', '')}_{cfg.get('partitioner', '')}"
    elif name.startswith("metrics_"):
        name = name[len("metrics_"):]

    return name, data


def load_results(files: list = None, search_dir: Path = None, load_all: bool = False) -> dict:
    """Load metrics results flexibly.

    Args:
        files: explicit list of file paths to load
        search_dir: directory to search in
        load_all: if True, load all metrics*.json in search_dir
    """
    results = {}

    if files:
        # Load explicitly specified files
        for f in files:
            p = Path(f)
            if not p.exists():
                print(f"  WARNING: {f} not found, skipping.")
                continue
            name, data = load_single_file(p)
            if data:
                results[name] = data

    elif load_all:
        # Load all metrics*.json in the directory
        search_dir = search_dir or Path(".")
        for f in sorted(search_dir.glob("metrics*.json")):
            name, data = load_single_file(f)
            if data:
                results[name] = data

    else:
        # Default: load metrics.json in current directory
        default = (search_dir or Path(".")) / "metrics.json"
        if default.exists():
            name, data = load_single_file(default)
            if data:
                results[name] = data

    return results


# ======================== Plot Functions ========================

def plot_accuracy_curves(results: dict, output_dir: Path, title_prefix: str = None):
    """Plot per-round accuracy curves for all experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        rounds = [r["round"] for r in data["per_round"]]
        accs = [r["accuracy"] for r in data["per_round"]]
        ax.plot(rounds, accs, marker="o", markersize=3, label=name)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Test Accuracy", fontsize=12)
    title = f"{title_prefix} - Accuracy vs. Round" if title_prefix else "Accuracy vs. Communication Round"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curves.png", dpi=150)
    print(f"  Saved: {output_dir / 'accuracy_curves.png'}")
    plt.close(fig)


def plot_loss_curves(results: dict, output_dir: Path, title_prefix: str = None):
    """Plot per-round loss curves for all experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        rounds = [r["round"] for r in data["per_round"]]
        losses = [r["loss"] for r in data["per_round"]]
        ax.plot(rounds, losses, marker="o", markersize=3, label=name)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Test Loss", fontsize=12)
    title = f"{title_prefix} - Loss vs. Round" if title_prefix else "Loss vs. Communication Round"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    print(f"  Saved: {output_dir / 'loss_curves.png'}")
    plt.close(fig)


def plot_accuracy_vs_communication(results: dict, output_dir: Path, title_prefix: str = None):
    """Plot accuracy as a function of cumulative communication cost (MB)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        comm = [r["cumulative_comm_mb"] for r in data["per_round"]]
        accs = [r["accuracy"] for r in data["per_round"]]
        ax.plot(comm, accs, marker="o", markersize=3, label=name)
    ax.set_xlabel("Cumulative Communication Cost (MB)", fontsize=12)
    ax.set_ylabel("Global Test Accuracy", fontsize=12)
    title = f"{title_prefix} - Accuracy vs. Comm Cost" if title_prefix else "Accuracy vs. Communication Cost"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_comm.png", dpi=150)
    print(f"  Saved: {output_dir / 'accuracy_vs_comm.png'}")
    plt.close(fig)


def plot_convergence_bar(results: dict, output_dir: Path, title_prefix: str = None):
    """Bar chart: rounds to reach target accuracies."""
    targets = [0.3, 0.5, 0.7, 0.8]
    names = list(results.keys())
    x = np.arange(len(targets))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(names):
        conv = results[name].get("convergence_targets", {})
        vals = []
        for t in targets:
            r = conv.get(str(t), conv.get(t, None))
            vals.append(r if r is not None else 0)
        bars = ax.bar(x + i * width, vals, width, label=name)
        for j, v in enumerate(vals):
            if v == 0:
                ax.text(x[j] + i * width, 0.5, "N/A", ha="center", va="bottom", fontsize=7, color="red")

    ax.set_xlabel("Target Accuracy", fontsize=12)
    ax.set_ylabel("Rounds to Reach", fontsize=12)
    title = f"{title_prefix} - Convergence Speed" if title_prefix else "Convergence Speed Comparison"
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"{int(t*100)}%" for t in targets])
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_bar.png", dpi=150)
    print(f"  Saved: {output_dir / 'convergence_bar.png'}")
    plt.close(fig)


def plot_final_comparison_bar(results: dict, output_dir: Path, title_prefix: str = None):
    """Bar chart comparing final accuracy, communication cost, and time."""
    names = list(results.keys())
    final_accs = [results[n]["final_accuracy"] for n in names]
    comm_mbs = [results[n]["total_communication_mb"] for n in names]
    total_times = [results[n]["total_time_sec"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].barh(names, final_accs, color="steelblue")
    axes[0].set_xlabel("Final Accuracy")
    axes[0].set_title("Final Accuracy")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(final_accs):
        axes[0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=8)

    axes[1].barh(names, comm_mbs, color="coral")
    axes[1].set_xlabel("Communication Cost (MB)")
    axes[1].set_title("Total Communication Cost")
    for i, v in enumerate(comm_mbs):
        axes[1].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    axes[2].barh(names, total_times, color="mediumseagreen")
    axes[2].set_xlabel("Total Time (s)")
    axes[2].set_title("Total Training Time")
    for i, v in enumerate(total_times):
        axes[2].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)

    suptitle = f"{title_prefix} - Comparison Summary" if title_prefix else "Experiment Comparison Summary"
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "final_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'final_comparison.png'}")
    plt.close(fig)


def plot_round_time(results: dict, output_dir: Path, title_prefix: str = None):
    """Plot per-round training time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        rounds = [r["round"] for r in data["per_round"]]
        times = [r["round_time_sec"] for r in data["per_round"]]
        ax.plot(rounds, times, marker="o", markersize=3, label=name)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Round Time (seconds)", fontsize=12)
    title = f"{title_prefix} - Round Time" if title_prefix else "Per-Round Training Time"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "round_time.png", dpi=150)
    print(f"  Saved: {output_dir / 'round_time.png'}")
    plt.close(fig)


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(
        description="Plot FL experiment results",
        epilog="""Examples:
  python plot_results.py                                    # plot metrics.json
  python plot_results.py metrics_fedprox.json               # plot one specific file
  python plot_results.py metrics_fedavg.json metrics_fedprox.json  # compare two
  python plot_results.py --all                              # all metrics*.json in .
  python plot_results.py --all --dir results                # all in results/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help="Specific metrics JSON file(s) to plot")
    parser.add_argument("--all", action="store_true", help="Load all metrics*.json files in the directory")
    parser.add_argument("--dir", type=str, default=".", help="Directory to search (default: current dir)")
    parser.add_argument("--output", type=str, default="", help="Output directory for plots (default: same as --dir)")
    parser.add_argument("--title", type=str, default="", help="Title prefix for all plots (e.g. 'Part1: IID Comparison')")
    args = parser.parse_args()

    search_dir = Path(args.dir)
    output_dir = Path(args.output) if args.output else search_dir
    output_dir.mkdir(exist_ok=True)

    # Load results based on arguments
    if args.files:
        results = load_results(files=args.files)
    elif args.all:
        results = load_results(search_dir=search_dir, load_all=True)
    else:
        results = load_results(search_dir=search_dir)

    if not results:
        print(f"No metrics files found. Run experiments first or specify files explicitly.")
        return

    print(f"Loaded {len(results)} experiment(s): {', '.join(results.keys())}")
    print(f"Generating plots in {output_dir.resolve()}...\n")

    title_prefix = args.title if args.title else None
    plot_accuracy_curves(results, output_dir, title_prefix)
    plot_loss_curves(results, output_dir, title_prefix)
    plot_accuracy_vs_communication(results, output_dir, title_prefix)
    if len(results) >= 2:
        plot_convergence_bar(results, output_dir, title_prefix)
        plot_final_comparison_bar(results, output_dir, title_prefix)
    plot_round_time(results, output_dir, title_prefix)

    print(f"\nAll plots saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
