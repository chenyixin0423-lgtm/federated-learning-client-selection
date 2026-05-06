"""Comprehensive evaluation metrics tracker for federated learning experiments.

Tracks the following metrics per round:
  - Accuracy (global test accuracy)
  - Loss (global test loss)
  - Convergence Speed (rounds to reach target accuracy, accuracy gain per round)
  - Communication Cost (cumulative model parameters transferred)
  - Per-round training time
  - Number of participating clients per round
"""

import json
import time
from pathlib import Path


class MetricsTracker:
    """Collects and persists evaluation metrics across FL rounds."""

    def __init__(self, num_model_params: int, config_summary: dict):
        self.num_model_params = num_model_params
        self.config_summary = config_summary

        # Per-round records
        self.rounds = []

        # Convergence tracking
        self.target_accuracies = [0.3, 0.5, 0.7, 0.8, 0.9]
        self.rounds_to_target = {}  # target -> first round that reached it

        # Communication cost tracking (in number of float32 parameters)
        self.cumulative_upload_params = 0  # clients -> server
        self.cumulative_download_params = 0  # server -> clients

        # Timing
        self._round_start = None

    def start_round(self):
        """Call at the beginning of each FL round."""
        self._round_start = time.time()

    def end_round(
        self,
        server_round: int,
        accuracy: float,
        loss: float,
        num_train_clients: int,
        num_eval_clients: int,
    ):
        """Record metrics at the end of an FL round.

        Communication cost model:
          - Download: server broadcasts 1 copy of the model to each training client
          - Upload: each training client sends 1 copy back
        """
        round_time = time.time() - self._round_start if self._round_start else 0.0

        # Communication cost for this round
        round_download = num_train_clients * self.num_model_params
        round_upload = num_train_clients * self.num_model_params
        self.cumulative_download_params += round_download
        self.cumulative_upload_params += round_upload
        cumulative_total = self.cumulative_download_params + self.cumulative_upload_params

        # Convergence: check if we just hit a target accuracy
        for target in self.target_accuracies:
            if target not in self.rounds_to_target and accuracy >= target:
                self.rounds_to_target[target] = server_round

        # Accuracy delta compared to previous round
        prev_acc = self.rounds[-1]["accuracy"] if self.rounds else 0.0
        acc_delta = accuracy - prev_acc

        record = {
            "round": server_round,
            "accuracy": accuracy,
            "loss": loss,
            "acc_delta": acc_delta,
            "num_train_clients": num_train_clients,
            "num_eval_clients": num_eval_clients,
            "round_time_sec": round(round_time, 2),
            "round_comm_params": round_download + round_upload,
            "round_comm_mb": round((round_download + round_upload) * 4 / (1024 ** 2), 2),
            "cumulative_comm_params": cumulative_total,
            "cumulative_comm_mb": round(cumulative_total * 4 / (1024 ** 2), 2),
        }
        self.rounds.append(record)

        # Print round summary
        print(
            f"  [Metrics] Round {server_round}: "
            f"acc={accuracy:.4f} (delta={acc_delta:+.4f}), "
            f"loss={loss:.4f}, "
            f"train_clients={num_train_clients}, "
            f"comm_cost={record['round_comm_mb']:.1f}MB (total={record['cumulative_comm_mb']:.1f}MB), "
            f"time={round_time:.1f}s"
        )

    def summary(self) -> dict:
        """Return a summary dict of the entire experiment."""
        if not self.rounds:
            return {}

        final = self.rounds[-1]
        best_round = max(self.rounds, key=lambda r: r["accuracy"])

        return {
            "config": self.config_summary,
            "total_rounds": len(self.rounds) - 1,  # exclude round 0 (baseline)
            "final_accuracy": final["accuracy"],
            "final_loss": final["loss"],
            "best_accuracy": best_round["accuracy"],
            "best_accuracy_round": best_round["round"],
            "convergence_targets": self.rounds_to_target,
            "total_communication_mb": final["cumulative_comm_mb"],
            "total_communication_params": final["cumulative_comm_params"],
            "avg_round_time_sec": round(
                sum(r["round_time_sec"] for r in self.rounds) / len(self.rounds), 2
            ),
            "total_time_sec": round(sum(r["round_time_sec"] for r in self.rounds), 2),
            "per_round": self.rounds,
        }

    def save(self, path: str = "metrics.json"):
        """Save metrics to a JSON file."""
        data = self.summary()
        Path(path).write_text(json.dumps(data, indent=2, default=str))
        print(f"\n[Metrics] Results saved to {path}")

    def print_summary(self):
        """Print a formatted experiment summary."""
        s = self.summary()
        if not s:
            print("[Metrics] No data collected.")
            return

        print("\n" + "=" * 60)
        print("  EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"  Strategy:         {s['config'].get('strategy', 'N/A')}")
        print(f"  Dataset:          {s['config'].get('dataset', 'N/A')}")
        print(f"  Model:            {s['config'].get('model', 'N/A')}")
        print(f"  Client Selection: {s['config'].get('client_selection', 'N/A')}")
        print(f"  Partitioner:      {s['config'].get('partitioner', 'N/A')}")
        print("-" * 60)
        print(f"  Total Rounds:     {s['total_rounds']}")
        print(f"  Final Accuracy:   {s['final_accuracy']:.4f}")
        print(f"  Best Accuracy:    {s['best_accuracy']:.4f} (round {s['best_accuracy_round']})")
        print(f"  Final Loss:       {s['final_loss']:.4f}")
        print("-" * 60)
        print("  Convergence (rounds to reach target accuracy):")
        for target in self.target_accuracies:
            r = self.rounds_to_target.get(target)
            status = f"round {r}" if r is not None else "not reached"
            print(f"    {target*100:.0f}% -> {status}")
        print("-" * 60)
        print(f"  Total Comm Cost:  {s['total_communication_mb']:.1f} MB")
        print(f"  Avg Round Time:   {s['avg_round_time_sec']:.2f} s")
        print(f"  Total Time:       {s['total_time_sec']:.2f} s")
        print("=" * 60)
