# RPL Discrete-Event Simulation

A Python/SimPy simulation of the **IPv6 Routing Protocol for Low-Power
and Lossy Networks** (RPL, RFC 6550) including the Trickle algorithm
(RFC 6206) and both OF0 and MRHOF objective functions.

---

## Project layout

```
rpl_sim/
  __init__.py       package init
  messages.py       DIS / DIO / DAO dataclasses (RFC 6550 §6)
  rpl.py            Objective Functions (OF0, MRHOF), neighbour table
  trickle.py        Trickle timer (RFC 6206)
  node.py           RPL node — SimPy process, DODAG state machine
  network.py        Radio medium, topology factories
  metrics.py        Metrics collection & CSV export
  visualize.py      DODAG graph & timeline plots (networkx + matplotlib)
experiments/
  sweep.py          Automated parameter sweeps
  plots.py          Standalone plotting from CSV
config.py           SimConfig dataclass with all defaults
main.py             CLI entry point
requirements.txt
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a default simulation (20 nodes, 300 s)
python main.py

# 4. Inspect outputs
#    results/summary.csv          — aggregate metrics
#    results/messages.csv         — per-message log
#    results/nodes.csv            — per-node log
#    results/dodag.png            — DODAG topology visualisation
#    results/message_timeline.png — DIS/DIO/DAO rate over time
```

---

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--nodes N` | 20 | Total node count (including root) |
| `--area F` | 100.0 | Deployment area side length (m) |
| `--range F` | 35.0 | Radio transmission range (m) |
| `--loss F` | 0.0 | Per-link packet loss probability [0, 1) |
| `--duration F` | 300.0 | Simulated time (seconds) |
| `--seed N` | 42 | Random seed |
| `--ocp {0,1}` | 0 | Objective Function: 0=OF0, 1=MRHOF |
| `--no-trickle` | — | Disable Trickle; use fixed 10 s DIO period |
| `--trickle-imin F` | 1.0 | Trickle Imin (s) |
| `--trickle-imax N` | 8 | Trickle doublings to Imax |
| `--trickle-k N` | 10 | Trickle redundancy constant |
| `--topology {random,grid}` | random | Placement strategy |
| `--rows / --cols / --spacing` | 4/5/25 | Grid topology parameters |
| `--output DIR` | results | Output directory |
| `--verbose` | — | DEBUG-level logging |

**Examples**

```bash
# 50 nodes, 10 % loss, MRHOF
python main.py --nodes 50 --loss 0.10 --ocp 1

# Grid, Trickle off
python main.py --topology grid --rows 5 --cols 6 --no-trickle

# Verbose debug output
python main.py --nodes 10 --verbose
```

---

## Parameter sweeps

```bash
# Full sweep (takes several minutes)
python -m experiments.sweep

# Quick subset (30–60 s)
python -m experiments.sweep --quick
```

Outputs to `results/sweep/`:
- `sweep_nodes.csv` / `sweep_nodes_plot.png`
- `sweep_loss.csv`
- `sweep_trickle.csv`

---

## Architecture decisions

### Route-over vs mesh-under
This simulation implements **route-over** architecture (RFC 6550 §3):
each node appears as a full IPv6 router hop, rank reflects hop distance,
and parent-selection is performed at the IP layer.  Under *mesh-under*,
the link layer would hide multi-hop paths behind a single IP hop, which
defeats the rank semantics used by RPL's objective functions.

### Addressing
Each node `k` carries the address `fd00::<k>/128` (ULA /128 host
route).  The root also aggregates `fd00::/64`.  DAOs advertise /128
prefixes upward so each DODAG ancestor builds a downward routing table
entry pointing to the next-hop child.

### Mode of Operation
MOP = 2 (Storing mode without multicast support).  Each non-root node
caches downward routes received via DAO and forwards DAOs with its own
prefix appended.  This enables bidirectional data flow without source
routing.

### Trickle
Trickle (RFC 6206) governs DIO transmission.  When the DODAG is
stable, the interval doubles to Imax, dramatically reducing DIO
overhead.  On an inconsistency (new neighbour, rank change), the timer
resets to Imin for rapid re-convergence.  Consistent DIO receptions
from neighbours suppress local transmission when `c ≥ k`.

---

## Performance metrics

| Metric | Definition |
|--------|------------|
| **Convergence time** | Time from DODAG root startup until the last node receives a parent and a finite rank. |
| **Control overhead** | Total DIS + DIO + DAO bytes transmitted across all nodes. |
| **PDR** | (TX attempts − lost packets) / TX attempts under the lossy channel model. |
| **Avg hop count** | Mean number of parent hops from non-root nodes to root. |
| **Energy proxy** | Total messages transmitted per node (proportional to radio-on time). |

---

## Running the test suite

```bash
pip install pytest
pytest tests/
```

---

## References

- RFC 6550 — RPL: IPv6 Routing Protocol for Low-Power and Lossy Networks
- RFC 6206 — The Trickle Algorithm
- RFC 6552 — Objective Function Zero for RPL (OF0)
- RFC 6719 — The Minimum Rank with Hysteresis Objective Function (MRHOF)
- SimPy documentation: https://simpy.readthedocs.io/
