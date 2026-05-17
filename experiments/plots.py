"""Standalone plotting script for sweep CSV results.

Usage:
    python -m experiments.plots results/sweep/sweep_nodes.csv  --out results/sweep
    python -m experiments.plots results/sweep/sweep_loss.csv   --loss --out results/sweep
    python -m experiments.plots results/sweep/sweep_trickle.csv --trickle --out results/sweep
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE = True
except ImportError:
    _HAVE = False


def _load(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(v: Optional[str]) -> Optional[float]:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def plot_node_sweep(csv_path: str, out_dir: str) -> None:
    if not _HAVE:
        return
    rows = _load(csv_path)
    x = [_f(r.get("num_nodes")) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Sweep: Node Count", fontsize=14)

    _line(axes[0, 0], x, [_f(r.get("convergence_time_s")) for r in rows],
          "Nodes", "Convergence time (s)", "Convergence Time")
    _line(axes[0, 1], x, [_f(r.get("total_control_bytes")) for r in rows],
          "Nodes", "Total bytes", "Control Message Overhead")
    _line(axes[1, 0], x, [_f(r.get("pdr")) for r in rows],
          "Nodes", "PDR", "Packet Delivery Ratio")
    _line(axes[1, 1], x, [_f(r.get("avg_hop_count")) for r in rows],
          "Nodes", "Avg hops", "Average Hop Count to Root")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, os.path.join(out_dir, "node_sweep.png"))


def plot_loss_sweep(csv_path: str, out_dir: str) -> None:
    if not _HAVE:
        return
    rows = _load(csv_path)
    x = [_f(r.get("loss_probability")) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Sweep: Packet Loss Probability", fontsize=14)

    _line(axes[0], x, [_f(r.get("pdr")) for r in rows],
          "Loss prob.", "PDR", "Packet Delivery Ratio", color="#e74c3c")
    _line(axes[1], x, [_f(r.get("convergence_time_s")) for r in rows],
          "Loss prob.", "Conv. time (s)", "Convergence Time", color="#2980b9")
    _line(axes[2], x, [_f(r.get("total_control_bytes")) for r in rows],
          "Loss prob.", "Total bytes", "Control Overhead", color="#27ae60")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, os.path.join(out_dir, "loss_sweep.png"))


def plot_trickle_sweep(csv_path: str, out_dir: str) -> None:
    if not _HAVE:
        return
    rows = _load(csv_path)

    on_rows = [r for r in rows if r.get("trickle") in ("True", "1", "true")]
    off_rows = [r for r in rows if r.get("trickle") in ("False", "0", "false")]

    x_on = [_f(r.get("num_nodes")) for r in on_rows]
    x_off = [_f(r.get("num_nodes")) for r in off_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Trickle ON vs OFF: DIO Overhead", fontsize=14)

    ax = axes[0]
    ax.plot(x_on, [_f(r.get("dio_count")) for r in on_rows],
            marker="o", label="Trickle ON", color="#2980b9")
    ax.plot(x_off, [_f(r.get("dio_count")) for r in off_rows],
            marker="s", label="Trickle OFF", color="#e74c3c")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("DIO messages sent")
    ax.set_title("DIO Count")
    ax.legend()
    ax.grid(True, linewidth=0.4)

    ax = axes[1]
    ax.plot(x_on, [_f(r.get("convergence_time_s")) for r in on_rows],
            marker="o", label="Trickle ON", color="#2980b9")
    ax.plot(x_off, [_f(r.get("convergence_time_s")) for r in off_rows],
            marker="s", label="Trickle OFF", color="#e74c3c")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Convergence time (s)")
    ax.set_title("Convergence Time")
    ax.legend()
    ax.grid(True, linewidth=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, os.path.join(out_dir, "trickle_sweep.png"))


def _line(ax, xs, ys, xlabel, ylabel, title, color="steelblue") -> None:
    clean = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if clean:
        cx, cy = zip(*clean)
        ax.plot(cx, cy, marker="o", color=color, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.4)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to sweep CSV file")
    parser.add_argument("--out", default="results/sweep")
    parser.add_argument("--loss", action="store_true", help="Loss sweep CSV")
    parser.add_argument("--trickle", action="store_true", help="Trickle sweep CSV")
    args = parser.parse_args()

    if args.loss:
        plot_loss_sweep(args.csv, args.out)
    elif args.trickle:
        plot_trickle_sweep(args.csv, args.out)
    else:
        plot_node_sweep(args.csv, args.out)
