# -*- coding: utf-8 -*-
"""Generate a clean hierarchical-tree DODAG figure for the baseline scenario."""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import simpy

from rpl_sim.metrics import MetricsCollector
from rpl_sim.network import create_random_network
from rpl_sim.rpl import create_objective_function

# ── rebuild the exact baseline network (seed 42, 20 nodes) ─────────────────
env = simpy.Environment()
metrics = MetricsCollector()
of = create_objective_function(0)
network = create_random_network(
    env=env, metrics=metrics, num_nodes=20, area_size=100.0, radio_range=35.0,
    loss_probability=0.0, objective_function=of, seed=42, use_trickle=True,
    trickle_imin=1.0, trickle_imax_doublings=8, trickle_k=10,
    ensure_connected=True)
for nid in network.nodes:
    metrics.register_node(nid)
for node in network.nodes.values():
    node.start()
env.run(until=300.0)

nodes = network.nodes
root_id = next(nid for nid, n in nodes.items() if n.is_root)

# ── build the tree from each node's final preferred parent ─────────────────
children = defaultdict(list)
for nid, n in nodes.items():
    if n.preferred_parent is not None:
        children[n.preferred_parent].append(nid)
for k in children:
    children[k].sort()

depth = {}


def set_depth(node, d):
    depth[node] = d
    for c in children.get(node, []):
        set_depth(c, d + 1)


set_depth(root_id, 0)
maxd = max(depth.values())

# tidy-tree x layout: leaves get successive slots, parents centre over kids
xpos = {}
cnt = [0.0]


def set_x(node):
    kids = children.get(node, [])
    if not kids:
        xpos[node] = cnt[0]
        cnt[0] += 1.0
    else:
        for c in kids:
            set_x(c)
        xpos[node] = sum(xpos[c] for c in kids) / len(kids)


set_x(root_id)
minx, maxx = min(xpos.values()), max(xpos.values())
xpos[root_id] = (minx + maxx) / 2.0   # centre the root above the whole tree
pos = {nid: (xpos[nid], -depth[nid]) for nid in nodes}

rank_at = {depth[nid]: nodes[nid].rank for nid in nodes}
tier_count = defaultdict(int)
for d in depth.values():
    tier_count[d] += 1

# ── draw ───────────────────────────────────────────────────────────────────
DPI = 150
fig, ax = plt.subplots(figsize=(14.0, 6.6), dpi=DPI)

NODE = "#3D74B8"
NODE_EDGE = "#24486f"
ROOT = "#C0392B"
ROOT_EDGE = "#641e16"
EDGE = "#9aa6b2"

# faint alternating tier bands + left-hand tier labels
for d in range(maxd + 1):
    y = -d
    ax.axhspan(y - 0.5, y + 0.5,
               facecolor=("#f2f5f9" if d % 2 else "#ffffff"), zorder=0)
    head = "Root" if d == 0 else (f"{d} hop" if d == 1 else f"{d} hops")
    ax.text(minx - 1.7, y, f"{head}\nrank {rank_at[d]}",
            ha="right", va="center", fontsize=9.5, fontweight="bold",
            color="#566573")
    ax.text(maxx + 0.55, y, f"{tier_count[d]} "
            + ("node" if tier_count[d] == 1 else "nodes"),
            ha="left", va="center", fontsize=8.5, style="italic",
            color="#909aa5")

# preferred-parent edges (child -> parent, arrowhead toward the root)
for nid, n in nodes.items():
    if n.preferred_parent is None:
        continue
    cx, cy = pos[nid]
    px, py = pos[n.preferred_parent]
    to_root = (n.preferred_parent == root_id)
    ann = ax.annotate(
        "", xy=(px, py), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.5,
                        shrinkA=16, shrinkB=25 if to_root else 17))
    ann.arrow_patch.set_zorder(2)

# non-root routing nodes
nr = [nid for nid in nodes if nid != root_id]
ax.scatter([pos[i][0] for i in nr], [pos[i][1] for i in nr],
           s=820, c=NODE, edgecolors=NODE_EDGE, linewidths=1.2, zorder=5)
for nid in nr:
    ax.text(*pos[nid], str(nid), ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=6)

# root node (apex of the tree)
rx, ry = pos[root_id]
ax.scatter([rx], [ry], s=1700, c=ROOT, edgecolors=ROOT_EDGE,
           linewidths=2.0, zorder=7)
ax.text(rx, ry, str(root_id), ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="white", zorder=8)
ax.text(rx, ry + 0.46, "ROOT  (DODAG root / LBR)", ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color=ROOT, zorder=8)

# legend
legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ROOT,
           markeredgecolor=ROOT_EDGE, markersize=15, label="DODAG root"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE,
           markeredgecolor=NODE_EDGE, markersize=12, label="Routing node"),
    Line2D([0], [0], color=EDGE, lw=1.6, marker=">", markersize=7,
           label="Preferred-parent edge (upward route to root)"),
]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          ncol=3, fontsize=9.5, frameon=False)

ax.set_title("RPL DODAG  —  Baseline 20-Node Topology",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlim(minx - 3.6, maxx + 1.9)
ax.set_ylim(-maxd - 0.78, 1.05)
ax.axis("off")
fig.tight_layout()
fig.savefig("results/dodag_hierarchical.png", dpi=DPI, bbox_inches="tight")
print("saved results/dodag_hierarchical.png")
print(f"tiers: {dict(sorted(tier_count.items()))}  maxdepth={maxd}")
