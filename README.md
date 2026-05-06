# Federated Learning Simulation Platform

A configurable federated learning simulation platform based on Flower and PyTorch. Supports multiple FL strategies (FedAvg, FedProx, FedAdagrad), datasets, models, client selection algorithms, and differential privacy, with comprehensive evaluation metrics and automated experiment tooling.

## Project Structure

```
quickstart-pytorch/
├── pytorchexample/
│   ├── __init__.py
│   ├── client_app.py        # ClientApp: local training & evaluation
│   ├── server_app.py        # ServerApp: orchestration & global evaluation
│   ├── task.py              # Model definitions, dataset loading, train/test
│   ├── custom_strategy.py   # Custom FL strategies & client selection algorithms
│   └── metrics_tracker.py   # Evaluation metrics: accuracy, convergence, comm cost
├── run_experiments.py        # Batch experiment runner (run all configs at once)
├── plot_results.py           # Visualization: generate comparison plots
├── pyproject.toml            # Dependencies & default run config
└── README.md
```

## Environment Setup

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-compatible GPU for acceleration

### Step 1: Create Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -e .
```

This will install all required packages defined in `pyproject.toml`:

| Package | Version | Purpose |
|---------|---------|---------|
| `flwr[simulation]` | >= 1.26.0 | Flower framework with simulation engine |
| `flwr-datasets[vision]` | >= 0.5.0 | Federated dataset partitioning |
| `torch` | 2.8.0 | PyTorch deep learning framework |
| `torchvision` | 0.23.0 | Pre-trained models (ResNet18) & transforms |
| `numpy` | latest | Numerical computing (K-Means clustering) |
| `matplotlib` | latest | Result visualization and plotting |

> If you need a different PyTorch version (e.g. CUDA-specific), install it first following https://pytorch.org/get-started/locally/, then run `pip install -e .`.

## How to Run

### Basic Run (default config)

```bash
flwr run .
```

This uses all default values from `pyproject.toml`: FedAvg + Simple CNN + CIFAR-10 + IID + Random selection.

### Run with Custom Config

Use `--run-config` to override any parameter at runtime:

```bash
flwr run . --run-config "key1=value1 key2=value2"
```

## Configurable Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `strategy` | `fedavg`, `fedprox`, `fedadagrad` | `fedavg` | FL aggregation strategy |
| `proximal-mu` | float | `0.1` | FedProx proximal term coefficient (only for `fedprox`) |
| `model` | `cnn`, `resnet18` | `cnn` | Model architecture |
| `dataset` | `cifar10`, `mnist`, `fashion-mnist` | `cifar10` | Training dataset |
| `partitioner` | `iid`, `dirichlet` | `iid` | Data partition method |
| `dirichlet-alpha` | float (e.g. 0.1 ~ 1.0) | `0.5` | Non-IID degree (lower = more heterogeneous) |
| `client-selection` | `random`, `high-loss`, `cluster-based`, `power-of-choice` | `random` | Client selection algorithm |
| `num-server-rounds` | int | `3` | Number of FL rounds |
| `learning-rate` | float | `0.1` | Learning rate |
| `local-epochs` | int | `1` | Local training epochs per round |
| `batch-size` | int | `32` | Batch size |
| `fraction-evaluate` | float (0~1) | `0.5` | Fraction of clients for evaluation |
| `franction-train` | float (0~1) | `0.5` | Fraction of clients for training |
| `dp-clip` | float (0 = off) | `0.0` | Differential privacy: gradient clipping max L2 norm |
| `dp-noise` | float (0 = off) | `0.0` | Differential privacy: Gaussian noise std |

## Example Experiments

### Single Experiment

```bash
# FedAvg + CIFAR-10 + IID
flwr run . --run-config "strategy='fedavg' dataset='cifar10' partitioner='iid' client-selection='random' model='cnn' num-server-rounds=20"

# FedProx + CIFAR-10 + Non-IID (Dirichlet)
flwr run . --run-config "strategy='fedprox' proximal-mu=0.1 dataset='cifar10' partitioner='dirichlet' dirichlet-alpha=0.3 client-selection='random' num-server-rounds=20"

# FedAvg + Non-IID + Cluster-based selection
flwr run . --run-config "strategy='fedavg' dataset='cifar10' partitioner='dirichlet' dirichlet-alpha=0.3 client-selection='cluster-based' num-server-rounds=20"

# FedAvg + Differential Privacy enabled
flwr run . --run-config "strategy='fedavg' dataset='cifar10' dp-clip=1.0 dp-noise=0.01 num-server-rounds=20"

# ResNet18 + MNIST + Non-IID
flwr run . --run-config "strategy='fedadagrad' dataset='mnist' model='resnet18' partitioner='dirichlet' dirichlet-alpha=0.5 client-selection='high-loss' num-server-rounds=20"
```

### Batch Experiments

Run all predefined experiment combinations at once:

```bash
# Run all experiments, save results to ./results/
python run_experiments.py

# Preview commands without running
python run_experiments.py --dry-run

# Run only experiments matching a keyword
python run_experiments.py --filter noniid

# Custom results directory
python run_experiments.py --results-dir ./my_results
```

The predefined experiments in `run_experiments.py` cover:
- Strategy comparison (FedAvg vs FedProx vs FedAdagrad) under IID and non-IID
- Client selection comparison (Random vs High-loss vs Cluster-based vs Power-of-choice)
- Model comparison (Simple CNN vs ResNet18)
- Dataset comparison (CIFAR-10 vs MNIST)
- Differential privacy impact

#### Compare Mode

Use `--compare` to fix a base config and only vary one parameter. This is the recommended way to run controlled experiments:

```bash
# Compare client selection strategies (fixed FedProx + CIFAR10 + Non-IID)
python run_experiments.py --compare client-selection --strategy fedprox --dataset cifar10 --partitioner dirichlet --dirichlet-alpha 0.3 --rounds 30

# Compare aggregation strategies (fixed CIFAR10 + Non-IID + Random)
python run_experiments.py --compare strategy --dataset cifar10 --partitioner dirichlet --dirichlet-alpha 0.3 --client-selection random --rounds 30

# Compare datasets (fixed FedAvg + IID + Random)
python run_experiments.py --compare dataset --strategy fedavg --partitioner iid --rounds 30

# Compare IID vs Non-IID (fixed FedAvg + CIFAR10)
python run_experiments.py --compare partitioner --strategy fedavg --dataset cifar10 --rounds 30

# Compare models (fixed MNIST + IID + FedAvg)
python run_experiments.py --compare model --strategy fedavg --dataset mnist --partitioner iid --rounds 20

# Only compare specific values
python run_experiments.py --compare client-selection --values "random,cluster-based" --strategy fedprox --dataset cifar10 --partitioner dirichlet --rounds 30

# Preview commands without running
python run_experiments.py --compare client-selection --strategy fedprox --dataset cifar10 --partitioner dirichlet --rounds 30 --dry-run
```

| Research Question | --compare | Fix everything else |
|-------------------|-----------|---------------------|
| Which client selection is best? | `client-selection` | strategy + dataset + partitioner |
| Which strategy is most stable under Non-IID? | `strategy` | dataset + selection + partitioner |
| How much does Non-IID hurt? | `partitioner` | strategy + dataset + selection |
| Which dataset is harder? | `dataset` | strategy + selection + partitioner |
| CNN vs ResNet18? | `model` | strategy + dataset + partitioner |

### Visualization

After running experiments, generate comparison plots:

```bash
# 1. Default: process metrics.json in current directory
python plot_results.py

# 2. Specify a single file
python plot_results.py metrics_fedprox_cifar10.json

# 3. Compare multiple experiments
python plot_results.py metrics_fedavg_cifar10.json metrics_fedprox_cifar10.json metrics_fedadagrad_cifar10.json

# 4. Compare all metrics*.json files in current directory
python plot_results.py --all

# 5. Specify output directory
python plot_results.py metrics_fedavg.json metrics_fedprox.json --output plots
```

Generated plots:
- `accuracy_curves.png` — accuracy vs. round for all experiments
- `loss_curves.png` — loss vs. round
- `accuracy_vs_comm.png` — accuracy vs. cumulative communication cost (MB)
- `convergence_bar.png` — rounds to reach 30%/50%/70%/80% accuracy
- `final_comparison.png` — side-by-side bar chart of final accuracy, comm cost, and time
- `round_time.png` — per-round training time

## Output

After each experiment:

- **Console**: prints per-round metrics and a final summary table
- **`metrics.json`**: full experiment results including:
  - Per-round accuracy, loss, accuracy delta
  - Convergence speed (rounds to reach 30%/50%/70%/80%/90% accuracy)
  - Communication cost per round and cumulative (in MB)
  - Round time and total time
  - Experiment config snapshot
- **`final_model.pt`**: saved final global model weights

## Client Count Configuration

Change the number of simulated clients in `pyproject.toml`:

```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 10  # number of simulated clients
```

## GPU Acceleration

If your system has a CUDA GPU, the code will automatically use it. To configure GPU resources for simulation, edit `pyproject.toml`:

```toml
[tool.flwr.federations.local-simulation.options]
backend.client-resources.num-gpus = 0.5  # GPU fraction per client
```

### If you encounter Ray-related errors, clean up and retry:

```toml
# Kill leftover Ray processes
taskkill /F /IM raylet.exe 2>$null

# Re-run the experiment
flwr run . --stream --run-config "..."
```

## Added:

1. FedProx Strategy (strategy="fedprox")

    Why it matters: FedProx is the standard solution for non-IID data heterogeneity in FL. It adds a proximal term (μ/2)·‖w - w_global‖² to each client's local loss, penalizing drift from the global model.

    Code: custom_strategy.py → CustomFedProx class injects proximal_mu into training config; task.py → train() computes the proximal term when proximal_mu > 0.
Config: strategy='fedprox' proximal-mu=0.1


2. Differential Privacy (dp-clip, dp-noise)

    Why it matters: FL alone doesn't guarantee privacy — gradients can leak information. Client-side DP (gradient clipping + Gaussian noise) provides formal privacy guarantees.

    Code: task.py → train() applies clip_grad_norm_ and noise injection after loss.backward().

    Config: dp-clip=1.0 dp-noise=0.01 (both 0 = disabled)


3. Batch Experiment Runner (run_experiments.py)

    13 predefined experiments covering strategy comparison, client selection comparison, model comparison, dataset comparison, and DP impact

    python run_experiments.py runs all; --dry-run previews; --filter selects subset
    
    Auto-saves each result as results/metrics_<name>.json and prints a comparison table


4. Visualization Utility (plot_results.py)
Generates 6 publication-ready plots from experiment results:

    Accuracy curves, loss curves, accuracy vs. communication cost
    
    Convergence speed bar chart, final comparison summary, per-round time