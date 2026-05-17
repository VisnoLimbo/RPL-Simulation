"""DODAG and metric visualisation utilities.

Requires networkx and matplotlib (listed in requirements.txt).
All functions accept an optional *save_path* to write a PNG and a
*show* flag for interactive display.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")   # headless backend; switch to "TkAgg" for interactive
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    _HAVE_PLOT = True
except ImportError:
    _HAVE_PLOT = False
    logger.warning("matplotlib / networkx not available; visualisation disabled")

if TYPE_CHECKING:
    from .network import Network
    from .metrics import MetricsCollector


def visualize_dodag(
    network: "Network",
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "RPL DODAG",
) -> None:
    """Draw the DODAG topology with nodes coloured by rank.

    Parameters
    ----------
    network:   Fully simulated Network instance.
    save_path: If given, save figure as PNG at this path.
    show:      If True, open an interactive window (requires a display).
    title:     Figure title string.
    """
    if not _HAVE_PLOT:
        logger.warning("visualize_dodag: matplotlib not available, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    nodes = network.nodes
    root_id = next((nid for nid, n in nodes.items() if n.is_root), None)

    # Build directed graph (child → parent edges)
    G = nx.DiGraph()
    for nid, node in nodes.items():
        G.add_node(nid)
        if node.preferred_parent is not None:
            G.add_edge(nid, node.preferred_parent)

    pos = {nid: node.position for nid, node in nodes.items()}

    # Colour by rank bucket
    from .messages import INFINITE_RANK, ROOT_RANK
    max_finite_rank = max(
        (n.rank for n in nodes.values() if n.rank < INFINITE_RANK),
        default=ROOT_RANK,
    )

    def rank_colour(rank: int) -> str:
        if rank == ROOT_RANK:
            return "#e74c3c"   # red = root
        if rank >= INFINITE_RANK:
            return "#95a5a6"   # grey = unassociated
        frac = min(rank / max_finite_rank, 1.0)
        # Gradient from gold (close to root) to steelblue (far)
        r = int(0xFF * (1 - frac) + 0x46 * frac)
        g = int(0xD7 * (1 - frac) + 0x82 * frac)
        b = int(0x00 * (1 - frac) + 0xB4 * frac)
        return f"#{r:02x}{g:02x}{b:02x}"

    node_colours = [rank_colour(nodes[nid].rank) for nid in G.nodes()]

    # Draw radio-range circles (faint)
    sample_range = next(iter(nodes.values())).radio_range
    for nid, node in nodes.items():
        circle = plt.Circle(
            node.position, sample_range,
            color="lightgrey", fill=False, linewidth=0.4, linestyle="--"
        )
        ax.add_patch(circle)

    # Draw all radio links (grey, thin)
    for nid in nodes:
        for nb in network.get_neighbors(nid):
            if nb > nid:
                x_vals = [pos[nid][0], pos[nb][0]]
                y_vals = [pos[nid][1], pos[nb][1]]
                ax.plot(x_vals, y_vals, color="#cccccc", linewidth=0.5, zorder=1)

    # Draw DODAG edges (parent links, blue arrows)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#2980b9",
        arrows=True,
        arrowsize=15,
        width=1.8,
        node_size=300,
        connectionstyle="arc3,rad=0.0",
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colours,
        node_size=300,
        linewidths=1.0,
        edgecolors="black",
    )

    # Labels: node_id (rank)
    labels = {
        nid: f"{nid}\n({n.rank})"
        for nid, n in nodes.items()
    }
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6)

    # Legend
    patches = [
        mpatches.Patch(color="#e74c3c", label="Root"),
        mpatches.Patch(color="#FFD700", label="Low rank (near root)"),
        mpatches.Patch(color="#4682B4", label="High rank (far from root)"),
        mpatches.Patch(color="#95a5a6", label="Unassociated"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"DODAG figure saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_message_timeline(
    metrics: "MetricsCollector",
    save_path: Optional[str] = None,
    show: bool = True,
    bin_width: float = 5.0,
) -> None:
    """Bar chart of control messages transmitted per time bin.

    Breaks down DIS / DIO / DAO per *bin_width*-second window.
    """
    if not _HAVE_PLOT:
        return

    events = metrics._msg_events
    if not events:
        logger.warning("plot_message_timeline: no message events recorded")
        return

    max_t = max(ev.time for ev in events)
    bins = [b * bin_width for b in range(int(max_t / bin_width) + 2)]

    counts: dict = {t: {"DIS": 0, "DIO": 0, "DAO": 0} for t in bins[:-1]}
    for ev in events:
        bucket = bins[int(ev.time / bin_width)]
        counts[bucket][ev.msg_type] += 1

    xs = list(counts.keys())
    dis_ys = [counts[x]["DIS"] for x in xs]
    dio_ys = [counts[x]["DIO"] for x in xs]
    dao_ys = [counts[x]["DAO"] for x in xs]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = bin_width * 0.28
    ax.bar(xs, dis_ys, width=width, label="DIS", color="#e67e22", align="edge")
    ax.bar([x + width for x in xs], dio_ys, width=width, label="DIO", color="#2980b9", align="edge")
    ax.bar([x + 2 * width for x in xs], dao_ys, width=width, label="DAO", color="#27ae60", align="edge")

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Messages transmitted")
    ax.set_title("RPL Control Message Timeline")
    ax.legend()
    ax.grid(True, axis="y", linewidth=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Timeline figure saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_sweep_results(
    csv_path: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Four-panel plot of sweep results loaded from a CSV file.

    Expected CSV columns (produced by experiments/sweep.py):
      num_nodes, convergence_time_s, total_control_bytes,
      pdr, avg_hop_count, [dio_count_no_trickle, dio_count_trickle]
    """
    if not _HAVE_PLOT:
        return

    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: _safe_float(v) for k, v in row.items()})

    if not rows:
        return

    def col(key: str) -> list:
        return [r[key] for r in rows if r.get(key) is not None]

    x = col("num_nodes")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("RPL Simulation — Parameter Sweep", fontsize=14)

    _plot_line(axes[0, 0], x, col("convergence_time_s"),
               "Nodes", "Convergence time (s)", "Convergence Time vs Node Count",
               color="#e74c3c")

    _plot_line(axes[0, 1], x, col("total_control_bytes"),
               "Nodes", "Total control bytes", "Control Overhead vs Node Count",
               color="#2980b9")

    _plot_line(axes[1, 0], x, col("pdr"),
               "Nodes", "PDR", "Packet Delivery Ratio vs Node Count",
               color="#27ae60")

    _plot_line(axes[1, 1], x, col("avg_hop_count"),
               "Nodes", "Avg hop count", "Average Hop Count vs Node Count",
               color="#8e44ad")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Sweep plot saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_line(ax, xs, ys, xlabel, ylabel, title, color="steelblue") -> None:
    ax.plot(xs, ys, marker="o", color=color, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.4)


def _safe_float(v: str) -> Optional[float]:
    try:
        return float(v) if v not in ("None", "", "nan") else None
    except (TypeError, ValueError):
        return None
