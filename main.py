"""RPL simulation entry point.

Usage examples
--------------
# Defaults (20 nodes, 300 s, Trickle on, OF0):
    python main.py

# 50 nodes, 10 % packet loss, MRHOF, verbose:
    python main.py --nodes 50 --loss 0.10 --ocp 1 --verbose

# Grid topology, no Trickle:
    python main.py --topology grid --rows 4 --cols 5 --no-trickle

# Save results to a custom directory:
    python main.py --output my_results
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import simpy

from config import SimConfig
from rpl_sim.metrics import MetricsCollector
from rpl_sim.network import create_random_network, create_grid_network
from rpl_sim.rpl import create_objective_function
from rpl_sim.visualize import visualize_dodag, plot_message_timeline


def setup_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_simulation(cfg: SimConfig) -> tuple:
    """Execute one simulation run and return (summary_dict, network, metrics)."""
    env = simpy.Environment()
    metrics = MetricsCollector()
    of = create_objective_function(cfg.ocp)

    trickle_kw = dict(
        use_trickle=cfg.use_trickle,
        trickle_imin=cfg.trickle_imin,
        trickle_imax_doublings=cfg.trickle_imax_doublings,
        trickle_k=cfg.trickle_k,
    )

    if cfg.topology == "grid":
        from rpl_sim.network import create_grid_network
        network = create_grid_network(
            env=env, metrics=metrics,
            rows=cfg.grid_rows, cols=cfg.grid_cols,
            spacing=cfg.grid_spacing,
            radio_range=cfg.radio_range,
            loss_probability=cfg.loss_probability,
            objective_function=of,
            seed=cfg.seed,
            **trickle_kw,
        )
    else:
        network = create_random_network(
            env=env, metrics=metrics,
            num_nodes=cfg.num_nodes,
            area_size=cfg.area_size,
            radio_range=cfg.radio_range,
            loss_probability=cfg.loss_probability,
            objective_function=of,
            seed=cfg.seed,
            ensure_connected=True,
            **trickle_kw,
        )

    total_nodes = len(network.nodes)

    # Register nodes with metrics collector
    for nid in network.nodes:
        metrics.register_node(nid)

    # Start all node SimPy processes
    for node in network.nodes.values():
        node.start()

    logging.info(
        f"Starting simulation: {total_nodes} nodes, "
        f"seed={cfg.seed}, "
        f"duration={cfg.sim_duration}s, "
        f"loss={cfg.loss_probability:.0%}, "
        f"trickle={'on' if cfg.use_trickle else 'off'}, "
        f"OF={'OF0' if cfg.ocp == 0 else 'MRHOF'}"
    )

    env.run(until=cfg.sim_duration)

    summary = metrics.summary(network.nodes)
    valid, msg = network.dodag_is_valid()
    summary["dodag_valid"] = valid
    summary["dodag_msg"] = msg

    logging.info("Simulation complete.")
    _print_summary(summary)

    return summary, network, metrics


def save_outputs(
    cfg: SimConfig,
    summary: dict,
    network,
    metrics: MetricsCollector,
) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    metrics.export_summary_csv(
        os.path.join(cfg.output_dir, "summary.csv"), network.nodes
    )
    metrics.export_message_log_csv(
        os.path.join(cfg.output_dir, "messages.csv")
    )
    metrics.export_node_log_csv(
        os.path.join(cfg.output_dir, "nodes.csv")
    )

    # DODAG visualisation
    dodag_path = os.path.join(cfg.output_dir, "dodag.png")
    visualize_dodag(network, save_path=dodag_path, show=False)

    # Message timeline plot
    timeline_path = os.path.join(cfg.output_dir, "message_timeline.png")
    plot_message_timeline(metrics, save_path=timeline_path, show=False)

    logging.info(f"All outputs written to '{cfg.output_dir}/'")


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 55)
    print("  RPL Simulation Results")
    print("=" * 55)
    width = max(len(k) for k in s)
    for k, v in s.items():
        if isinstance(v, float) and v is not None:
            print(f"  {k:<{width}}  {v:.4f}")
        else:
            print(f"  {k:<{width}}  {v}")
    print("=" * 55 + "\n")


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="RPL Discrete-Event Simulation")
    p.add_argument("--nodes", type=int, default=20)
    p.add_argument("--area", type=float, default=100.0)
    p.add_argument("--range", dest="radio_range", type=float, default=35.0)
    p.add_argument("--loss", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=random.randrange(1_000_000),
                   help="Random seed (default: random each run; "
                        "pass a fixed value to reproduce a topology)")
    p.add_argument("--ocp", type=int, choices=[0, 1], default=0,
                   help="Objective Code Point: 0=OF0, 1=MRHOF")
    p.add_argument("--no-trickle", action="store_true",
                   help="Disable Trickle algorithm (fixed DIO interval)")
    p.add_argument("--trickle-imin", type=float, default=1.0)
    p.add_argument("--trickle-imax", type=int, default=8,
                   help="Number of doublings for Imax")
    p.add_argument("--trickle-k", type=int, default=10)
    p.add_argument("--topology", choices=["random", "grid"], default="random")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--spacing", type=float, default=25.0)
    p.add_argument("--output", default="results")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    return SimConfig(
        num_nodes=args.nodes,
        area_size=args.area,
        radio_range=args.radio_range,
        loss_probability=args.loss,
        sim_duration=args.duration,
        seed=args.seed,
        ocp=args.ocp,
        use_trickle=not args.no_trickle,
        trickle_imin=args.trickle_imin,
        trickle_imax_doublings=args.trickle_imax,
        trickle_k=args.trickle_k,
        topology=args.topology,
        grid_rows=args.rows,
        grid_cols=args.cols,
        grid_spacing=args.spacing,
        output_dir=args.output,
        log_level="DEBUG" if args.verbose else "INFO",
    )


if __name__ == "__main__":
    cfg = parse_args()
    setup_logging(cfg.log_level)
    summary, network, metrics = run_simulation(cfg)
    save_outputs(cfg, summary, network, metrics)
