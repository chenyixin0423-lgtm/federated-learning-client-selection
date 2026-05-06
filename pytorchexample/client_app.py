"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import get_model, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Read dataset and model config
    dataset_name = str(context.run_config.get("dataset", "cifar10")).lower()
    model_name = str(context.run_config.get("model", "cnn")).lower()
    partitioner_type = str(context.run_config.get("partitioner", "iid")).lower()
    alpha = float(context.run_config.get("dirichlet-alpha", 0.5))

    # Load model matching the dataset and model choice
    model = get_model(dataset_name, model_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size,
        dataset_name=dataset_name,
        partitioner_type=partitioner_type,
        alpha=alpha,
    )

    # Read FedProx and DP config
    proximal_mu = float(msg.content["config"].get("proximal_mu", 0.0))
    dp_clip = float(context.run_config.get("dp-clip", 0.0))
    dp_noise = float(context.run_config.get("dp-noise", 0.0))

    # Save a copy of global params for FedProx proximal term
    global_params = [p.clone().detach() for p in model.parameters()] if proximal_mu > 0 else None

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        proximal_mu=proximal_mu,
        global_params=global_params,
        dp_clip=dp_clip,
        dp_noise=dp_noise,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Read dataset and model config
    dataset_name = str(context.run_config.get("dataset", "cifar10")).lower()
    model_name = str(context.run_config.get("model", "cnn")).lower()
    partitioner_type = str(context.run_config.get("partitioner", "iid")).lower()
    alpha = float(context.run_config.get("dirichlet-alpha", 0.5))

    # Load model matching the dataset and model choice
    model = get_model(dataset_name, model_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(
        partition_id, num_partitions, batch_size,
        dataset_name=dataset_name,
        partitioner_type=partitioner_type,
        alpha=alpha,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
