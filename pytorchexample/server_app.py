"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdagrad

from pytorchexample.task import get_model, load_centralized_dataset, test
from pytorchexample.custom_strategy import CustomFedAvg, CustomFedAdagrad, CustomFedProx
from pytorchexample.metrics_tracker import MetricsTracker

# Create ServerApp
app = ServerApp()


def build_strategy(context: Context):
    """Build strategy based on run config.

    Config keys:
        strategy: "fedavg" | "fedadagrad" (default: "fedavg")
        client-selection: "random" | "high-loss" | "power-of-choice" | "cluster-based"
        fraction-evaluate: float
        franction-train: float
    """
    VALID_STRATEGIES = {"fedavg", "fedprox", "fedadagrad"}
    VALID_SELECTIONS = {"random", "high-loss", "cluster-based", "power-of-choice"}

    strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()
    client_selection = str(context.run_config.get("client-selection", "random")).lower()

    if strategy_name not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Must be one of: {', '.join(sorted(VALID_STRATEGIES))}"
        )
    if client_selection not in VALID_SELECTIONS:
        raise ValueError(
            f"Unknown client-selection '{client_selection}'. Must be one of: {', '.join(sorted(VALID_SELECTIONS))}"
        )

    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    fraction_train = float(context.run_config["franction-train"])

    common_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        client_selection=client_selection,
    )

    if strategy_name == "fedadagrad":
        strategy = CustomFedAdagrad(**common_kwargs)
        print(f"Using strategy: FedAdagrad, client selection: {client_selection}")
    elif strategy_name == "fedprox":
        proximal_mu = float(context.run_config.get("proximal-mu", 0.1))
        strategy = CustomFedProx(proximal_mu=proximal_mu, **common_kwargs)
        print(f"Using strategy: FedProx (mu={proximal_mu}), client selection: {client_selection}")
    else:
        strategy = CustomFedAvg(**common_kwargs)
        print(f"Using strategy: FedAvg, client selection: {client_selection}")

    return strategy


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    VALID_DATASETS = {"cifar10", "mnist", "fashion-mnist"}
    VALID_MODELS = {"cnn", "resnet18"}

    dataset_name: str = str(context.run_config.get("dataset", "cifar10")).lower()
    model_name: str = str(context.run_config.get("model", "cnn")).lower()

    if dataset_name not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Must be one of: {', '.join(sorted(VALID_DATASETS))}"
        )
    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Must be one of: {', '.join(sorted(VALID_MODELS))}"
        )

    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f"Num rounds: {num_rounds}")

    # Load global model (matching dataset + model choice)
    global_model = get_model(dataset_name, model_name)
    arrays = ArrayRecord(global_model.state_dict())

    # Count model parameters for communication cost tracking
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build strategy from config
    strategy = build_strategy(context)

    # Initialize metrics tracker
    config_summary = {
        "strategy": str(context.run_config.get("strategy", "fedavg")),
        "dataset": dataset_name,
        "model": model_name,
        "client_selection": str(context.run_config.get("client-selection", "random")),
        "partitioner": str(context.run_config.get("partitioner", "iid")),
        "dirichlet_alpha": float(context.run_config.get("dirichlet-alpha", 0.5)),
        "num_rounds": num_rounds,
        "learning_rate": lr,
        "batch_size": int(context.run_config.get("batch-size", 32)),
        "local_epochs": int(context.run_config.get("local-epochs", 1)),
        "fraction_train": float(context.run_config.get("franction-train", 0.5)),
        "fraction_evaluate": float(context.run_config.get("fraction-evaluate", 0.5)),
        "proximal_mu": float(context.run_config.get("proximal-mu", 0.0)),
        "dp_clip": float(context.run_config.get("dp-clip", 0.0)),
        "dp_noise": float(context.run_config.get("dp-noise", 0.0)),
    }
    tracker = MetricsTracker(num_model_params=num_params, config_summary=config_summary)

    # Define evaluate function with the correct dataset and metrics tracking
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = get_model(dataset_name, model_name)
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_dataloader = load_centralized_dataset(dataset_name)
        test_loss, test_acc = test(model, test_dataloader, device)

        # Record metrics
        # Estimate participating client counts from fraction config
        num_nodes = len(list(grid.get_node_ids()))
        frac_train = float(context.run_config.get("franction-train", 0.5))
        frac_eval = float(context.run_config.get("fraction-evaluate", 0.5))
        num_train_clients = max(1, int(num_nodes * frac_train))
        num_eval_clients = max(1, int(num_nodes * frac_eval))

        tracker.end_round(
            server_round=server_round,
            accuracy=test_acc,
            loss=test_loss,
            num_train_clients=num_train_clients,
            num_eval_clients=num_eval_clients,
        )
        tracker.start_round()  # Start timing next round

        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    # Start timing first round
    tracker.start_round()

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Print and save comprehensive metrics
    tracker.print_summary()
    tracker.save("metrics.json")

    # Save final model
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
