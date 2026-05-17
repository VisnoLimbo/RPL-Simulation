"""RPL Objective Functions and per-neighbour state.

Two OFs are implemented:

OF0  (RFC 6552) – hop-count metric.  Rank = parent_rank + 3×256.
     Simple and loop-free.  OCP = 0x0000.

MRHOF (RFC 6719) – ETX-weighted metric.  Rank reflects expected
      link cost.  Includes hysteresis to avoid oscillation.
      OCP = 0x0001.

Architecture note:  Route-over with IPv6 /128 host-routes is assumed
throughout.  Mesh-under would hide the hop structure from the IP layer
and is incompatible with the RFC 6550 rank semantics used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from .messages import INFINITE_RANK, MIN_HOP_RANK_INCREASE, ROOT_RANK, DEFAULT_STEP_OF_RANK

# ── Objective Code Points ──────────────────────────────────────────────────────
OCP_OF0: int = 0x0000
OCP_MRHOF: int = 0x0001


class NodeState(Enum):
    """RPL node state machine (RFC 6550 §8.2)."""
    UNASSOCIATED = auto()   # No DODAG, no parent
    ASSOCIATED = auto()     # Member of a DODAG, has a preferred parent
    ROOT = auto()           # DODAG root (LBR)


@dataclass
class NeighborEntry:
    """One row in the node's neighbour table.

    Populated / updated on receipt of each DIO from that neighbour.
    """
    node_id: int
    rank: int = INFINITE_RANK
    dodag_id: str = ""
    dodag_version: int = 0
    dtsn: int = 0
    last_heard: float = 0.0
    link_metric: float = 1.0   # ETX or hop-count proxy
    is_preferred_parent: bool = False


# ── Objective Function base ────────────────────────────────────────────────────

class ObjectiveFunction:
    """Abstract base for RPL Objective Functions."""

    OCP: int = OCP_OF0

    def rank_increase(self, link_metric: float) -> int:
        """Rank increment contributed by one hop with given link metric."""
        raise NotImplementedError

    def compute_rank(self, parent_rank: int, link_metric: float) -> int:
        """Node rank given parent rank and link metric to that parent."""
        raise NotImplementedError

    def best_parent(self, candidates: List[NeighborEntry]) -> Optional[NeighborEntry]:
        """Select preferred parent from a list of candidate neighbours."""
        raise NotImplementedError


# ── OF0 ────────────────────────────────────────────────────────────────────────

class OF0(ObjectiveFunction):
    """Objective Function Zero (RFC 6552).

    Pure hop-count: each hop contributes DEFAULT_STEP_OF_RANK × 256 = 768.
    Parent is chosen by minimum rank; ties broken by node_id (deterministic).
    """

    OCP = OCP_OF0

    def rank_increase(self, link_metric: float = 1.0) -> int:
        return DEFAULT_STEP_OF_RANK * MIN_HOP_RANK_INCREASE  # 768

    def compute_rank(self, parent_rank: int, link_metric: float = 1.0) -> int:
        if parent_rank >= INFINITE_RANK:
            return INFINITE_RANK
        new_rank = parent_rank + self.rank_increase(link_metric)
        return min(new_rank, INFINITE_RANK)

    def best_parent(self, candidates: List[NeighborEntry]) -> Optional[NeighborEntry]:
        valid = [n for n in candidates if 0 < n.rank < INFINITE_RANK]
        if not valid:
            return None
        # Deterministic tie-breaking ensures reproducibility
        return min(valid, key=lambda n: (n.rank, n.node_id))


# ── MRHOF ──────────────────────────────────────────────────────────────────────

class MRHOF(ObjectiveFunction):
    """Minimum Rank with Hysteresis Objective Function (RFC 6719).

    Uses ETX as the path metric.  ETX is carried in link_metric
    (1.0 for a perfect link; >1.0 for a lossy link).

    Hysteresis: the current parent is only replaced if the candidate
    offers a rank improvement of at least RANK_THRESHOLD.
    """

    OCP = OCP_MRHOF

    RANK_THRESHOLD = MIN_HOP_RANK_INCREASE // 2   # 128: ~half a hop improvement
    MAX_LINK_METRIC = 3.0                          # Discard links with ETX > 3

    def rank_increase(self, link_metric: float) -> int:
        return max(MIN_HOP_RANK_INCREASE, int(link_metric * MIN_HOP_RANK_INCREASE))

    def compute_rank(self, parent_rank: int, link_metric: float) -> int:
        if parent_rank >= INFINITE_RANK:
            return INFINITE_RANK
        new_rank = parent_rank + self.rank_increase(link_metric)
        return min(new_rank, INFINITE_RANK)

    def best_parent(
        self,
        candidates: List[NeighborEntry],
        current_parent_id: Optional[int] = None,
        current_rank: int = INFINITE_RANK,
    ) -> Optional[NeighborEntry]:
        valid = [
            n for n in candidates
            if 0 < n.rank < INFINITE_RANK and n.link_metric <= self.MAX_LINK_METRIC
        ]
        if not valid:
            return None

        def path_rank(n: NeighborEntry) -> int:
            return n.rank + self.rank_increase(n.link_metric)

        best = min(valid, key=lambda n: (path_rank(n), n.node_id))

        # Hysteresis: stick with current parent unless improvement is significant
        if current_parent_id is not None:
            current = next((n for n in valid if n.node_id == current_parent_id), None)
            if current is not None:
                if path_rank(current) - path_rank(best) < self.RANK_THRESHOLD:
                    return current

        return best


# ── Factory ────────────────────────────────────────────────────────────────────

def create_objective_function(ocp: int) -> ObjectiveFunction:
    """Instantiate the correct OF for the given Objective Code Point."""
    return {OCP_OF0: OF0, OCP_MRHOF: MRHOF}.get(ocp, OF0)()
