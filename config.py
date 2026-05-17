"""Default simulation configuration.

All parameters can be overridden from the command line (main.py) or
programmatically when calling run_simulation() directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimConfig:
    """Holds every tunable simulation parameter."""

    # ── Topology ──────────────────────────────────────────────────────────────
    num_nodes: int = 20
    area_size: float = 100.0       # Square area side length (metres)
    radio_range: float = 35.0      # Transmission range (metres)
    topology: str = "random"       # "random" | "grid"
    grid_rows: int = 4             # Used only when topology="grid"
    grid_cols: int = 5             # Used only when topology="grid"
    grid_spacing: float = 25.0     # Grid node spacing (metres)

    # ── Channel model ─────────────────────────────────────────────────────────
    loss_probability: float = 0.0  # Per-link Bernoulli loss [0, 1)

    # ── RPL protocol ─────────────────────────────────────────────────────────
    ocp: int = 0                   # Objective Code Point: 0=OF0, 1=MRHOF

    # ── Trickle ───────────────────────────────────────────────────────────────
    use_trickle: bool = True
    trickle_imin: float = 1.0      # Minimum interval (simulation seconds)
    trickle_imax_doublings: int = 8
    trickle_k: int = 10            # Redundancy constant

    # ── Simulation ────────────────────────────────────────────────────────────
    sim_duration: float = 300.0    # Total simulated time (seconds)
    seed: int = 42

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "results"
    log_level: str = "INFO"        # DEBUG | INFO | WARNING


# Sensible defaults used by main.py
DEFAULT_CONFIG = SimConfig()
