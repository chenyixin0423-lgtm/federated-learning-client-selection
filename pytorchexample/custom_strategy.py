"""Custom federated learning strategies with configurable client selection."""

import random
import numpy as np
from collections import defaultdict
from typing import Iterable

from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedAdagrad


# ======================== Client Selection Functions ========================

def _compute_num_to_select(grid: Grid, fraction: float, min_nodes: int = 1) -> tuple:
    """Compute how many nodes to select and return (node_ids, num_to_select)."""
    node_ids = list(grid.get_node_ids())
    num_to_select = max(min_nodes, int(len(node_ids) * fraction))
    num_to_select = min(num_to_select, len(node_ids))
    return node_ids, num_to_select


def select_clients_random(grid: Grid, fraction: float, min_nodes: int = 1) -> list:
    """Random client selection."""
    node_ids, num_to_select = _compute_num_to_select(grid, fraction, min_nodes)
    return random.sample(node_ids, num_to_select)


def select_clients_high_loss(
    grid: Grid, fraction: float, client_losses: dict, min_nodes: int = 1
) -> list:
    """High-loss client selection. Prioritizes clients with highest previous-round loss.

    Clients with no recorded loss are assigned a high default value (inf)
    so they get explored first, preventing the "frozen selection" problem
    where only initially-selected clients are ever chosen.
    """
    node_ids, num_to_select = _compute_num_to_select(grid, fraction, min_nodes)
    if not client_losses:
        return random.sample(node_ids, num_to_select)
    # Unknown clients get inf loss → explored first, then sorted by actual loss
    sorted_nodes = sorted(
        node_ids, key=lambda nid: client_losses.get(nid, float('inf')), reverse=True
    )
    return sorted_nodes[:num_to_select]


def select_clients_cluster_based(
    grid: Grid, fraction: float, client_metrics: dict,
    num_clusters: int = 3, min_nodes: int = 1,
    exploration_ratio: float = 0.3,
) -> list:
    """Cluster-based client selection. Groups by (loss, num_examples), samples per cluster.

    Uses exploration_ratio to balance between:
      - Exploitation: sample from clusters of known clients (diversity-aware)
      - Exploration: sample unknown clients to expand coverage

    This prevents the "frozen selection" problem where only a small set of
    initially-selected clients are ever chosen.
    """
    node_ids, num_to_select = _compute_num_to_select(grid, fraction, min_nodes)
    if not client_metrics:
        return random.sample(node_ids, num_to_select)

    nodes_with_metrics = [nid for nid in node_ids if nid in client_metrics]
    nodes_without_metrics = [nid for nid in node_ids if nid not in client_metrics]

    # Exploration: always reserve some slots for unseen clients
    num_explore = 0
    if nodes_without_metrics:
        num_explore = max(1, int(num_to_select * exploration_ratio))
        num_explore = min(num_explore, len(nodes_without_metrics), num_to_select - 1)
    num_exploit = num_to_select - num_explore

    # Exploitation: cluster-based selection from known clients
    selected = []
    if len(nodes_with_metrics) >= num_clusters and num_exploit > 0:
        features = np.array([
            [client_metrics[nid].get("loss", 0.0), client_metrics[nid].get("num_examples", 0)]
            for nid in nodes_with_metrics
        ])
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        denom = f_max - f_min
        denom[denom == 0] = 1.0
        features_norm = (features - f_min) / denom

        actual_k = min(num_clusters, len(nodes_with_metrics))
        labels = _kmeans(features_norm, actual_k)

        clusters = defaultdict(list)
        for i, nid in enumerate(nodes_with_metrics):
            clusters[labels[i]].append(nid)

        remaining_budget = num_exploit
        cluster_ids = sorted(clusters.keys())
        for idx, cid in enumerate(cluster_ids):
            members = clusters[cid]
            if idx == len(cluster_ids) - 1:
                n_from_cluster = remaining_budget
            else:
                n_from_cluster = max(1, int(num_exploit * len(members) / len(nodes_with_metrics)))
            n_from_cluster = min(n_from_cluster, len(members), remaining_budget)
            selected.extend(random.sample(members, n_from_cluster))
            remaining_budget -= n_from_cluster
            if remaining_budget <= 0:
                break
    else:
        # Not enough data for clustering, random from known
        selected = random.sample(nodes_with_metrics, min(num_exploit, len(nodes_with_metrics)))

    # Exploration: add unseen clients
    if num_explore > 0 and nodes_without_metrics:
        selected.extend(random.sample(nodes_without_metrics, num_explore))

    return selected[:num_to_select]


def _kmeans(X: np.ndarray, k: int, max_iter: int = 20) -> list:
    """Lightweight K-Means with K-Means++ init."""
    n = len(X)
    indices = [random.randint(0, n - 1)]
    for _ in range(1, k):
        dists = np.min([np.sum((X - X[c]) ** 2, axis=1) for c in indices], axis=0)
        probs = dists / (dists.sum() + 1e-12)
        indices.append(np.random.choice(n, p=probs))
    centroids = X[indices].copy()
    labels = [0] * n
    for _ in range(max_iter):
        for i in range(n):
            dists = [np.sum((X[i] - centroids[j]) ** 2) for j in range(k)]
            labels[i] = int(np.argmin(dists))
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)
        for i in range(n):
            new_centroids[labels[i]] += X[i]
            counts[labels[i]] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                new_centroids[j] = centroids[j]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels


def select_clients_power_of_choice(
    grid: Grid, fraction: float, client_losses: dict, d: int = 5, min_nodes: int = 1
) -> list:
    """Power-of-d-choice: sample d candidates per slot, pick highest loss.

    d controls the exploration-exploitation tradeoff:
      - d=1: pure random
      - d=5: moderate preference for high-loss (default)
      - d=total_clients: equivalent to pure high-loss

    Unknown clients get inf loss to ensure exploration.
    """
    node_ids, num_to_select = _compute_num_to_select(grid, fraction, min_nodes)
    if not client_losses or len(node_ids) <= num_to_select:
        return random.sample(node_ids, num_to_select)
    selected = []
    remaining = list(node_ids)
    for _ in range(num_to_select):
        candidates = random.sample(remaining, min(d, len(remaining)))
        best = max(candidates, key=lambda nid: client_losses.get(nid, float('inf')))
        selected.append(best)
        remaining.remove(best)
    return selected


# ======================== Mixin: Client Selection + Metrics Tracking ========================

class ClientSelectionMixin:
    """Mixin that provides custom client selection and per-client metrics tracking.

    Overrides configure_train to use custom selection, and aggregate_train to
    extract per-client losses from reply messages.
    """

    # Cluster cache for recluster_interval
    _cached_clusters = None
    _last_recluster_round = -1
    recluster_interval = 5  # recluster every 5 rounds

    def _select_nodes(self, grid: Grid, server_round: int = 0) -> list:
        """Select nodes based on self.client_selection strategy."""
        fraction = self.fraction_train
        selection = self.client_selection

        if selection == "high-loss":
            nodes = select_clients_high_loss(grid, fraction, self.client_losses)
        elif selection == "cluster-based":
            nodes = self._select_cluster_with_cache(grid, fraction, server_round)
        elif selection == "power-of-choice":
            nodes = select_clients_power_of_choice(grid, fraction, self.client_losses)
        else:
            nodes = select_clients_random(grid, fraction)

        print(f"  [Selection] {selection}: selected {len(nodes)} / {len(list(grid.get_node_ids()))} clients")
        return nodes

    def _select_cluster_with_cache(self, grid: Grid, fraction: float, server_round: int) -> list:
        """Cluster-based selection with recluster interval.

        Only recomputes clusters every `recluster_interval` rounds.
        Between reclusters, reuses cached cluster assignments but
        re-samples from each cluster for diversity.
        """
        need_recluster = (
            self._cached_clusters is None
            or server_round - self._last_recluster_round >= self.recluster_interval
        )

        if need_recluster:
            # Full recluster
            nodes = select_clients_cluster_based(grid, fraction, self.client_metrics)
            # Cache the cluster assignments for reuse
            self._rebuild_cluster_cache(grid)
            self._last_recluster_round = server_round
            print(f"    [Cluster] Reclustered at round {server_round}")
            return nodes
        else:
            # Reuse cached clusters, re-sample from each
            return self._sample_from_cached_clusters(grid, fraction)

    def _rebuild_cluster_cache(self, grid: Grid):
        """Rebuild and cache cluster assignments from current client_metrics."""
        node_ids = list(grid.get_node_ids())
        nodes_with_metrics = [nid for nid in node_ids if nid in self.client_metrics]

        if len(nodes_with_metrics) < 3:
            self._cached_clusters = None
            return

        features = np.array([
            [self.client_metrics[nid].get("loss", 0.0),
             self.client_metrics[nid].get("num_examples", 0)]
            for nid in nodes_with_metrics
        ])
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        denom = f_max - f_min
        denom[denom == 0] = 1.0
        features_norm = (features - f_min) / denom

        actual_k = min(3, len(nodes_with_metrics))
        labels = _kmeans(features_norm, actual_k)

        clusters = defaultdict(list)
        for i, nid in enumerate(nodes_with_metrics):
            clusters[labels[i]].append(nid)
        self._cached_clusters = dict(clusters)

    def _sample_from_cached_clusters(self, grid: Grid, fraction: float) -> list:
        """Sample from cached clusters without reclustering."""
        node_ids, num_to_select = _compute_num_to_select(grid, fraction)
        nodes_without_metrics = [nid for nid in node_ids if nid not in self.client_metrics]

        if not self._cached_clusters:
            return random.sample(node_ids, num_to_select)

        # Exploration slots
        num_explore = 0
        if nodes_without_metrics:
            num_explore = max(1, int(num_to_select * 0.3))
            num_explore = min(num_explore, len(nodes_without_metrics), num_to_select - 1)
        num_exploit = num_to_select - num_explore

        # Sample from cached clusters
        selected = []
        remaining_budget = num_exploit
        cluster_ids = sorted(self._cached_clusters.keys())
        total_in_clusters = sum(len(v) for v in self._cached_clusters.values())

        for idx, cid in enumerate(cluster_ids):
            members = self._cached_clusters[cid]
            if idx == len(cluster_ids) - 1:
                n = remaining_budget
            else:
                n = max(1, int(num_exploit * len(members) / total_in_clusters))
            n = min(n, len(members), remaining_budget)
            selected.extend(random.sample(members, n))
            remaining_budget -= n
            if remaining_budget <= 0:
                break

        # Exploration
        if num_explore > 0 and nodes_without_metrics:
            selected.extend(random.sample(nodes_without_metrics, num_explore))

        return selected[:num_to_select]

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Override: select clients with custom strategy, then build messages."""
        if self.fraction_train == 0.0:
            return []

        # Store current arrays for FedOpt/FedAdagrad aggregation
        # (FedOpt.configure_train normally does this, but we bypass it)
        if hasattr(self, 'current_arrays'):
            self.current_arrays = {k: array.numpy() for k, array in arrays.items()}

        # Pre-configure hook (for subclass-specific logic like LR decay)
        self._pre_configure_train(server_round, config)

        # Custom client selection
        node_ids = self._select_nodes(grid, server_round)

        # Inject server round into config
        config["server-round"] = server_round

        # Build messages for selected nodes
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple:
        """Override: extract per-client metrics, then call parent aggregation."""
        # Extract per-client losses from replies before aggregation
        reply_list = list(replies)
        for msg in reply_list:
            if msg.has_error():
                continue
            node_id = msg.metadata.src_node_id
            content = msg.content
            if "metrics" in content:
                metrics = content["metrics"]
                loss = float(metrics.get("train_loss", 0.0))
                num_examples = int(metrics.get("num-examples", 0))
                self.client_losses[node_id] = loss
                self.client_metrics[node_id] = {
                    "loss": loss,
                    "num_examples": num_examples,
                }

        print(f"  [Metrics] Tracked losses for {len(self.client_losses)} clients")

        # Call parent aggregation
        return super().aggregate_train(server_round, reply_list)

    def _pre_configure_train(self, server_round: int, config: ConfigRecord):
        """Hook for subclass-specific pre-configuration. Override in subclasses."""
        pass


# ======================== Strategy: Custom FedAvg ========================

class CustomFedAvg(ClientSelectionMixin, FedAvg):
    """FedAvg with configurable client selection strategy."""

    def __init__(self, client_selection: str = "random", **kwargs):
        super().__init__(**kwargs)
        self.client_selection = client_selection
        self.client_losses = {}
        self.client_metrics = {}

    def _pre_configure_train(self, server_round: int, config: ConfigRecord):
        print(f"[Round {server_round}] Strategy=FedAvg, ClientSelection={self.client_selection}")


# ======================== Strategy: Custom FedProx ========================

class CustomFedProx(ClientSelectionMixin, FedAvg):
    """FedProx: FedAvg + proximal term for non-IID robustness.

    Reference: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020).
    """

    def __init__(self, client_selection: str = "random", proximal_mu: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.client_selection = client_selection
        self.client_losses = {}
        self.client_metrics = {}
        self.proximal_mu = proximal_mu

    def _pre_configure_train(self, server_round: int, config: ConfigRecord):
        config["proximal_mu"] = self.proximal_mu
        print(f"[Round {server_round}] Strategy=FedProx (mu={self.proximal_mu}), ClientSelection={self.client_selection}")


# ======================== Strategy: Custom FedAdagrad ========================

class CustomFedAdagrad(ClientSelectionMixin, FedAdagrad):
    """FedAdagrad with LR decay and configurable client selection."""

    def __init__(
        self,
        client_selection: str = "random",
        lr_decay_interval: int = 5,
        lr_decay_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client_selection = client_selection
        self.client_losses = {}
        self.client_metrics = {}
        self.lr_decay_interval = lr_decay_interval
        self.lr_decay_factor = lr_decay_factor

    def _pre_configure_train(self, server_round: int, config: ConfigRecord):
        if server_round % self.lr_decay_interval == 0 and server_round > 0:
            config["lr"] *= self.lr_decay_factor
            print(f"[Round {server_round}] LR decreased to: {config['lr']}")
        print(f"[Round {server_round}] Strategy=FedAdagrad, ClientSelection={self.client_selection}")
