"""Batch experiment runner for federated learning simulations.

Runs a predefined set of experiments with different combinations of
strategies, datasets, models, partitioners, and client selection algorithms.
Each experiment produces a metrics_<name>.json file. After all experiments
finish, a comparison summary is printed.

Usage:
    python run_experiments.py                 # run all experiments
    python run_experiments.py --dry-run       # print commands without running
    python run_experiments.py --filter fedprox # only run experiments containing "fedprox"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ======================== Experiment Definitions ========================

EXPERIMENTS = [
    # --- Baseline comparisons: Strategy × IID ---
    {
        "name": "fedavg_cifar10_iid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedprox_cifar10_iid_random",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "proximal-mu": 0.1,
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedadagrad_cifar10_iid_random",
        "config": {
            "strategy": "fedadagrad",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },

    # --- Non-IID comparisons: Strategy × Dirichlet ---
    {
        "name": "fedavg_cifar10_noniid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedprox_cifar10_noniid_random",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "random",
            "proximal-mu": 0.1,
            "num-server-rounds": 20,
        },
    },

    # --- Client selection comparisons: FedAvg + non-IID ---
    {
        "name": "fedavg_cifar10_noniid_highloss",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "high-loss",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedavg_cifar10_noniid_cluster",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "cluster-based",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedavg_cifar10_noniid_poc",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "power-of-choice",
            "num-server-rounds": 20,
        },
    },

    # --- Client selection comparisons: FedProx + non-IID ---
    {
        "name": "fedprox_noniid_random",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "random",
            "proximal-mu": 0.1,
            "num-server-rounds": 30,
        },
    },
    {
        "name": "fedprox_noniid_highloss",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "high-loss",
            "proximal-mu": 0.1,
            "num-server-rounds": 30,
        },
    },
    {
        "name": "fedprox_noniid_cluster",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "cluster-based",
            "proximal-mu": 0.1,
            "num-server-rounds": 30,
        },
    },
    {
        "name": "fedprox_noniid_poc",
        "config": {
            "strategy": "fedprox",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "power-of-choice",
            "proximal-mu": 0.1,
            "num-server-rounds": 30,
        },
    },

    # --- Model comparison ---
    {
        "name": "fedavg_cifar10_iid_resnet18",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "resnet18",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },

    # --- Dataset comparison ---
    {
        "name": "fedavg_mnist_iid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "mnist",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },
    {
        "name": "fedavg_mnist_noniid_random",
        "config": {
            "strategy": "fedavg",
            "dataset": "mnist",
            "model": "cnn",
            "partitioner": "dirichlet",
            "dirichlet-alpha": 0.3,
            "client-selection": "random",
            "num-server-rounds": 20,
        },
    },

    # --- Differential Privacy ---
    {
        "name": "fedavg_cifar10_iid_dp",
        "config": {
            "strategy": "fedavg",
            "dataset": "cifar10",
            "model": "cnn",
            "partitioner": "iid",
            "client-selection": "random",
            "dp-clip": 1.0,
            "dp-noise": 0.01,
            "num-server-rounds": 20,
        },
    },
]


def build_run_config_str(config: dict) -> str:
    """Convert a config dict to a flwr --run-config string."""
    parts = []
    for k, v in config.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def _run_cmd(cmd: str) -> str:
    """Run a shell command and return combined stdout+stderr as UTF-8 string."""
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(
        cmd, shell=True, capture_output=True, env=env,
    )
    # Decode as UTF-8, ignore emoji/special chars that fail
    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    return stdout + stderr


def wait_for_run(run_id: str, poll_interval: int = 10, timeout: int = 3600):
    """Wait for a flwr run to finish by polling `flwr ls`."""
    print(f"  Waiting for run {run_id} to finish...")
    elapsed = 0
    while elapsed < timeout:
        output = _run_cmd("flwr ls")
        for line in output.splitlines():
            if run_id in line and "finished" in line:
                print(f"  Run {run_id} finished.")
                return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    print(f"  WARNING: Run {run_id} timed out after {timeout}s.")
    return False


def run_experiment(name: str, config: dict, results_dir: Path, dry_run: bool = False):
    """Run a single experiment and wait for it to complete."""
    config_str = build_run_config_str(config)
    cmd = f'flwr run . --run-config "{config_str}"'

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Command: {cmd}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Skipped.")
        return None

    # Start the run and capture the run ID
    output = _run_cmd(cmd)

    # Extract run ID from output like "Successfully started run 1234567890"
    run_id = None
    for line in output.splitlines():
        if "started run" in line.lower():
            # Find the numeric run ID at the end
            parts = line.strip().split()
            for part in reversed(parts):
                if part.isdigit():
                    run_id = part
                    break
            if run_id:
                print(f"  Started run ID: {run_id}")
                break

    if not run_id:
        print(f"  WARNING: Could not extract run ID from output:")
        print(f"  {output.strip()}")
        return None

    # Wait for the run to finish
    if not wait_for_run(run_id):
        return None

    # Give a moment for metrics.json to be written
    time.sleep(3)

    # Move metrics.json to results dir with experiment name
    metrics_src = Path("metrics.json")
    if metrics_src.exists():
        dest = results_dir / f"metrics_{name}.json"
        shutil.move(str(metrics_src), str(dest))
        print(f"  Metrics saved to: {dest}")
        return dest
    else:
        print(f"  WARNING: metrics.json not found for {name}")
        return None


def print_comparison(results_dir: Path):
    """Print a comparison table of all experiment results."""
    files = sorted(results_dir.glob("metrics_*.json"))
    if not files:
        print("\nNo results to compare.")
        return

    print(f"\n{'='*100}")
    print("  EXPERIMENT COMPARISON")
    print(f"{'='*100}")
    header = f"{'Experiment':<40} {'Accuracy':>10} {'Best Acc':>10} {'Loss':>10} {'Comm(MB)':>10} {'Time(s)':>10}"
    print(header)
    print("-" * 100)

    for f in files:
        data = json.loads(f.read_text())
        name = f.stem.replace("metrics_", "")
        final_acc = data.get("final_accuracy", 0)
        best_acc = data.get("best_accuracy", 0)
        final_loss = data.get("final_loss", 0)
        comm_mb = data.get("total_communication_mb", 0)
        total_time = data.get("total_time_sec", 0)
        print(f"{name:<40} {final_acc:>10.4f} {best_acc:>10.4f} {final_loss:>10.4f} {comm_mb:>10.1f} {total_time:>10.1f}")

    print(f"{'='*100}")

    # Convergence comparison
    print(f"\n{'='*80}")
    print("  CONVERGENCE COMPARISON (rounds to reach target accuracy)")
    print(f"{'='*80}")
    targets = ["0.3", "0.5", "0.7", "0.8", "0.9"]
    header = f"{'Experiment':<40} " + " ".join(f"{t:>8}%" for t in targets)
    print(header)
    print("-" * 80)

    for f in files:
        data = json.loads(f.read_text())
        name = f.stem.replace("metrics_", "")
        conv = data.get("convergence_targets", {})
        vals = []
        for t in targets:
            r = conv.get(t, conv.get(float(t), None))
            vals.append(f"{r:>8}" if r is not None else f"{'N/A':>8}")
        print(f"{name:<40} " + " ".join(vals))

    print(f"{'='*80}")


def generate_compare_experiments(base_config: dict, compare_key: str, compare_values: list) -> list:
    """Generate experiments by varying one parameter while keeping others fixed.

    Example:
        base = {"strategy": "fedprox", "dataset": "cifar10", ...}
        compare_key = "client-selection"
        compare_values = ["random", "high-loss", "cluster-based", "power-of-choice"]
    """
    experiments = []
    for val in compare_values:
        config = dict(base_config)
        config[compare_key] = val
        # Build a short name from the varying parameter
        val_short = str(val).replace("-", "")
        name = f"{config.get('strategy', 'exp')}_{config.get('dataset', '')}_{compare_key}_{val_short}"
        experiments.append({"name": name, "config": config})
    return experiments


# Default values for each compare dimension
COMPARE_DEFAULTS = {
    "client-selection": ["random", "high-loss", "cluster-based", "power-of-choice"],
    "strategy": ["fedavg", "fedprox", "fedadagrad"],
    "dataset": ["cifar10", "mnist", "fashion-mnist"],
    "model": ["cnn", "resnet18"],
    "partitioner": ["iid", "dirichlet"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run batch FL experiments",
        epilog="""Examples:
  python run_experiments.py                           # run all predefined experiments
  python run_experiments.py --filter fedprox          # only predefined exps matching "fedprox"
  python run_experiments.py --dry-run                 # preview without running

  # Compare mode: fix base config, vary one parameter
  python run_experiments.py --compare client-selection --strategy fedprox --dataset cifar10 --model cnn --partitioner dirichlet --dirichlet-alpha 0.3 --rounds 30
  python run_experiments.py --compare strategy --dataset cifar10 --model cnn --partitioner iid --rounds 20
  python run_experiments.py --compare dataset --strategy fedavg --model cnn --partitioner iid --rounds 20
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--filter", type=str, default="", help="Only run predefined experiments matching this string")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store results")

    # Compare mode arguments
    parser.add_argument("--compare", type=str, default="", metavar="PARAM",
                        help="Parameter to compare: client-selection, strategy, dataset, model, partitioner")
    parser.add_argument("--values", type=str, default="", help="Comma-separated values to compare (default: all valid values)")
    parser.add_argument("--strategy", type=str, default="fedavg")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--partitioner", type=str, default="iid")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--client-selection", type=str, default="random")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--proximal-mu", type=float, default=0.1)

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    if args.compare:
        # Compare mode: generate experiments dynamically
        base_config = {
            "strategy": args.strategy,
            "dataset": args.dataset,
            "model": args.model,
            "partitioner": args.partitioner,
            "dirichlet-alpha": args.dirichlet_alpha,
            "client-selection": args.client_selection,
            "num-server-rounds": args.rounds,
            "learning-rate": args.lr,
            "local-epochs": args.local_epochs,
            "proximal-mu": args.proximal_mu,
        }

        if args.values:
            compare_values = [v.strip() for v in args.values.split(",")]
        elif args.compare in COMPARE_DEFAULTS:
            compare_values = COMPARE_DEFAULTS[args.compare]
        else:
            print(f"Unknown compare parameter '{args.compare}'. Choose from: {', '.join(COMPARE_DEFAULTS.keys())}")
            return

        experiments = generate_compare_experiments(base_config, args.compare, compare_values)
        print(f"Compare mode: varying '{args.compare}' = {compare_values}")
        print(f"Base config: {', '.join(f'{k}={v}' for k, v in base_config.items() if k != args.compare)}")
    else:
        # Predefined experiments mode
        experiments = EXPERIMENTS
        if args.filter:
            experiments = [e for e in experiments if args.filter.lower() in e["name"].lower()]

    print(f"\nRunning {len(experiments)} experiment(s)...")
    print(f"Results directory: {results_dir.resolve()}")

    for exp in experiments:
        run_experiment(exp["name"], exp["config"], results_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print_comparison(results_dir)


if __name__ == "__main__":
    main()
