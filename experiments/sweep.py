"""Parameter sweep experiments for RPL simulation.

Sweeps:
  1. Node count (10 → 80) with fixed area / radio range.
  2. Packet loss probability (0 % → 40 %).
  3. Trickle ON vs OFF — DIO overhead comparison.

Each sweep point runs *REPEATS* independent seeds and averages results.
Outputs CSV files in results/sweep/ and a combined plot.

Usage:
    python -m experiments.sweep            # full sweep (slow)
    python -m experiments.sweep --quick    # fast subset
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from typing import List, Dict, Any

# Make sure the project root is on sys.path when run as __main__
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import simpy

from config import SimConfig
from main import run_simulation
from rpl_sim.visualize import plot_sweep_results

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

REPEATS = 3              # Seeds per sweep point
SIM_DURATION = 300.0     # Simulation horizon (s)
BASE_AREA = 100.0
BASE_RANGE = 35.0
OUTPUT_DIR = "results/sweep"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg(rows: List[Dict[str, Any]], key: str) -> Any:
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] != "None"]
    if not vals:
        return None
    try:
        return sum(float(v) for v in vals) / len(vals)
    except (TypeError, ValueError):
        return vals[0]


def _run_repeated(cfg: SimConfig, repeats: int = REPEATS) -> Dict[str, Any]:
    """Run *repeats* replicates and return mean metrics."""
    rows = []
    for i in range(repeats):
        cfg.seed = 42 + i * 7
        summary, _, _ = run_simulation(cfg)
        rows.append(summary)
    return {k: _avg(rows, k) for k in rows[0]}


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Sweep results → {path}")


# ── Sweep 1: node count ────────────────────────────────────────────────────────

def sweep_node_count(
    node_counts: List[int],
    repeats: int = REPEATS,
) -> List[Dict]:
    results = []
    for n in node_counts:
        logger.info(f"Sweep nodes: n={n}")
        cfg = SimConfig(
            num_nodes=n,
            area_size=BASE_AREA,
            radio_range=BASE_RANGE,
            loss_probability=0.0,
            sim_duration=SIM_DURATION,
            use_trickle=True,
            log_level="WARNING",
        )
        row = _run_repeated(cfg, repeats)
        row["num_nodes"] = n
        results.append(row)
    return results


# ── Sweep 2: loss probability ──────────────────────────────────────────────────

def sweep_loss(
    loss_probs: List[float],
    num_nodes: int = 20,
    repeats: int = REPEATS,
) -> List[Dict]:
    results = []
    for loss in loss_probs:
        logger.info(f"Sweep loss: p={loss:.2f}")
        cfg = SimConfig(
            num_nodes=num_nodes,
            area_size=BASE_AREA,
            radio_range=BASE_RANGE,
            loss_probability=loss,
            sim_duration=SIM_DURATION,
            use_trickle=True,
            log_level="WARNING",
        )
        row = _run_repeated(cfg, repeats)
        row["loss_probability"] = loss
        results.append(row)
    return results


# ── Sweep 3: Trickle ON vs OFF ────────────────────────────────────────────────

def sweep_trickle(
    node_counts: List[int],
    repeats: int = REPEATS,
) -> List[Dict]:
    results = []
    for n in node_counts:
        for trickle in (True, False):
            logger.info(f"Sweep trickle: n={n}, trickle={'on' if trickle else 'off'}")
            cfg = SimConfig(
                num_nodes=n,
                area_size=BASE_AREA,
                radio_range=BASE_RANGE,
                loss_probability=0.0,
                sim_duration=SIM_DURATION,
                use_trickle=trickle,
                log_level="WARNING",
            )
            row = _run_repeated(cfg, repeats)
            row["num_nodes"] = n
            row["trickle"] = trickle
            results.append(row)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main(quick: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.perf_counter()

    if quick:
        node_counts = [5, 10, 20, 30]
        loss_probs = [0.0, 0.1, 0.2]
    else:
        node_counts = [5, 10, 15, 20, 30, 40, 60, 80]
        loss_probs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

    # Sweep 1
    s1 = sweep_node_count(node_counts)
    write_csv(os.path.join(OUTPUT_DIR, "sweep_nodes.csv"), s1)

    # Sweep 2
    s2 = sweep_loss(loss_probs)
    write_csv(os.path.join(OUTPUT_DIR, "sweep_loss.csv"), s2)

    # Sweep 3
    s3 = sweep_trickle(node_counts[:4] if quick else node_counts[:6])
    write_csv(os.path.join(OUTPUT_DIR, "sweep_trickle.csv"), s3)

    # Plot sweep 1 (node count)
    try:
        plot_sweep_results(
            os.path.join(OUTPUT_DIR, "sweep_nodes.csv"),
            save_path=os.path.join(OUTPUT_DIR, "sweep_nodes_plot.png"),
            show=False,
        )
    except Exception as exc:
        logger.warning(f"Could not generate sweep plot: {exc}")

    elapsed = time.perf_counter() - t0
    logger.info(f"All sweeps complete in {elapsed:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run a fast subset of sweep points")
    args = parser.parse_args()
    main(quick=args.quick)
