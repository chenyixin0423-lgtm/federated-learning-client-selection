"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.models import resnet18


# ======================== Model Definitions ========================

class CifarNet(nn.Module):
    """Simple CNN for CIFAR-10 (3-channel 32x32 images, 10 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MnistNet(nn.Module):
    """Simple CNN for MNIST / Fashion-MNIST (1-channel 28x28 images, 10 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _create_resnet18(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """Create a ResNet18 adapted for small images (32x32 / 28x28).

    Modifications vs. standard ImageNet ResNet18:
      - Replace first 7x7 conv with 3x3 conv (better for small images)
      - Remove initial max-pool layer
      - Adjust fc output to num_classes
      - If in_channels=1, adapt first conv layer
    """
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # Replace all BatchNorm layers with track_running_stats=False.
    # This removes num_batches_tracked (int64) from state_dict, which is
    # incompatible with Flower's ArrayRecord/aggregation strategies.
    # Also recommended in FL since running stats are local per-client.
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=False,
            )
            # Set the new module on the parent
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_bn)
    return model


# ======================== Dataset Config ========================

DATASET_CONFIG = {
    "cifar10": {
        "hf_name": "uoft-cs/cifar10",
        "img_key": "img",
        "label_key": "label",
        "model_cls": CifarNet,
        "transforms": Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    },
    "mnist": {
        "hf_name": "ylecun/mnist",
        "img_key": "image",
        "label_key": "label",
        "model_cls": MnistNet,
        "transforms": Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
    },
    "fashion-mnist": {
        "hf_name": "zalando-datasets/fashion_mnist",
        "img_key": "image",
        "label_key": "label",
        "model_cls": MnistNet,
        "transforms": Compose([ToTensor(), Normalize((0.2860,), (0.3530,))]),
    },
}

# Keep backward compatibility
Net = CifarNet

fds = None  # Cache FederatedDataset
_current_dataset = None


def get_model(dataset_name: str = "cifar10", model_name: str = "cnn") -> nn.Module:
    """Return the appropriate model for the given dataset and model config.

    Args:
        dataset_name: "cifar10" | "mnist" | "fashion-mnist"
        model_name: "cnn" (simple CNN) | "resnet18"
    """
    cfg = DATASET_CONFIG[dataset_name]
    if model_name == "resnet18":
        in_channels = 3 if dataset_name == "cifar10" else 1
        return _create_resnet18(in_channels=in_channels, num_classes=10)
    return cfg["model_cls"]()


def _make_apply_transforms(dataset_name: str):
    """Create a transform function for the given dataset."""
    cfg = DATASET_CONFIG[dataset_name]
    img_key = cfg["img_key"]
    transforms = cfg["transforms"]

    def apply_transforms(batch):
        batch[img_key] = [transforms(img) for img in batch[img_key]]
        return batch

    return apply_transforms


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_name: str = "cifar10",
    partitioner_type: str = "iid",
    alpha: float = 0.5,
):
    """Load partitioned data for the given dataset.

    Args:
        partitioner_type: "iid" or "dirichlet"
        alpha: Dirichlet concentration parameter (lower = more non-IID)
    """
    global fds, _current_dataset
    cfg = DATASET_CONFIG[dataset_name]

    if fds is None or _current_dataset != dataset_name:
        if partitioner_type == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by=cfg["label_key"],
                alpha=alpha,
            )
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=cfg["hf_name"],
            partitioners={"train": partitioner},
        )
        _current_dataset = dataset_name

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    apply_fn = _make_apply_transforms(dataset_name)
    partition_train_test = partition_train_test.with_transform(apply_fn)

    img_key = cfg["img_key"]
    label_key = cfg["label_key"]

    def collate_fn(batch):
        images = torch.stack([item[img_key] for item in batch])
        labels = torch.tensor([item[label_key] for item in batch])
        return {"img": images, "label": labels}

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size,
        collate_fn=collate_fn,
    )
    return trainloader, testloader


def load_centralized_dataset(dataset_name: str = "cifar10"):
    """Load the centralized test set and return a dataloader."""
    cfg = DATASET_CONFIG[dataset_name]
    test_dataset = load_dataset(cfg["hf_name"], split="test")
    apply_fn = _make_apply_transforms(dataset_name)
    dataset = test_dataset.with_format("torch").with_transform(apply_fn)

    img_key = cfg["img_key"]
    label_key = cfg["label_key"]

    def collate_fn(batch):
        images = torch.stack([item[img_key] for item in batch])
        labels = torch.tensor([item[label_key] for item in batch])
        return {"img": images, "label": labels}

    return DataLoader(dataset, batch_size=128, collate_fn=collate_fn)


def train(net, trainloader, epochs, lr, device,
          proximal_mu: float = 0.0, global_params=None,
          dp_clip: float = 0.0, dp_noise: float = 0.0):
    """Train the model on the training set.

    Args:
        proximal_mu: FedProx proximal term coefficient. When > 0, adds
            (mu/2) * ||w - w_global||^2 to the loss to prevent client
            drift from the global model. Set to 0 for standard FedAvg.
        global_params: list of global model parameter tensors (required if proximal_mu > 0).
        dp_clip: Max L2 norm for per-sample gradient clipping (0 = disabled).
        dp_noise: Gaussian noise std added to clipped gradients (0 = disabled).
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)

            # FedProx: add proximal term (mu/2)*||w - w_global||^2
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for local_p, global_p in zip(net.parameters(), global_params):
                    proximal_term += ((local_p - global_p) ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * proximal_term

            loss.backward()

            # Differential Privacy: gradient clipping + noise injection
            if dp_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), dp_clip)
            if dp_noise > 0.0:
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * dp_noise

            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
