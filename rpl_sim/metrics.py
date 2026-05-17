"""Performance metrics collection and reporting.

Records per-event telemetry during the simulation and computes
aggregate statistics at the end:

  - DODAG convergence time
  - Control overhead (counts + bytes, per message type)
  - Packet delivery ratio
  - Average hop count to root
  - Per-node energy proxy (messages transmitted)
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MessageEvent:
    """Immutable record of a single transmitted control message."""
    time: float
    sender_id: int
    msg_type: str        # "DIS" | "DIO" | "DAO"
    size_bytes: int


@dataclass
class NodeRecord:
    """Mutable per-node state tracked by the collector."""
    node_id: int
    join_time: float = float("inf")
    parent_id: Optional[int] = None
    final_rank: int = 0
    sent: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recv: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bytes_sent: int = 0


class MetricsCollector:
    """Central store for all simulation metrics.

    All *record_* methods are designed to be called from within SimPy
    processes at the moment the event occurs; they are intentionally
    side-effect-only (no return value).
    """

    def __init__(self) -> None:
        self.dodag_start_time: float = 0.0
        self.dodag_id: str = ""

        self._nodes: Dict[int, NodeRecord] = {}
        self._msg_events: List[MessageEvent] = []

        # PDR tracking: each unicast/broadcast attempt counts as one TX attempt
        self.tx_attempts: int = 0
        self.tx_lost: int = 0

        # Route records (for verifying DAO coverage)
        self._routes: List[Tuple[float, int, List[str]]] = []

    # ── Registration ──────────────────────────────────────────────────────────

    def register_node(self, node_id: int) -> None:
        self._nodes[node_id] = NodeRecord(node_id=node_id)

    # ── Event recording ───────────────────────────────────────────────────────

    def record_dodag_start(self, time: float, dodag_id: str) -> None:
        self.dodag_start_time = time
        self.dodag_id = dodag_id

    def record_node_joined(
        self, time: float, node_id: int, parent_id: int, rank: int
    ) -> None:
        rec = self._nodes.get(node_id)
        if rec and rec.join_time == float("inf"):
            rec.join_time = time
            rec.parent_id = parent_id
            rec.final_rank = rank

    def record_message_sent(
        self, time: float, node_id: int, msg_type: str, size_bytes: int
    ) -> None:
        self._msg_events.append(MessageEvent(time, node_id, msg_type, size_bytes))
        rec = self._nodes.get(node_id)
        if rec:
            rec.sent[msg_type] += 1
            rec.bytes_sent += size_bytes

    def record_message_received(self, node_id: int, msg_type: str) -> None:
        rec = self._nodes.get(node_id)
        if rec:
            rec.recv[msg_type] += 1

    def record_tx_attempt(self) -> None:
        """One unicast or one broadcast-to-neighbour attempt."""
        self.tx_attempts += 1

    def record_packet_loss(self) -> None:
        self.tx_lost += 1

    def record_route_installed(
        self, time: float, from_node: int, prefixes: List[str]
    ) -> None:
        self._routes.append((time, from_node, prefixes))

    # ── Derived statistics ────────────────────────────────────────────────────

    def convergence_time(self, root_id: int = 0) -> Optional[float]:
        """Time (s) from DODAG start until ALL non-root nodes have joined.

        Returns None if some nodes have not yet joined.
        """
        non_root = [r for nid, r in self._nodes.items() if nid != root_id]
        if not non_root:
            return 0.0
        if any(r.join_time == float("inf") for r in non_root):
            return None
        return max(r.join_time for r in non_root) - self.dodag_start_time

    def message_overhead(self) -> Dict[str, Dict[str, int]]:
        """Aggregate message counts and byte totals per message type."""
        counts: Dict[str, int] = defaultdict(int)
        totals: Dict[str, int] = defaultdict(int)
        for ev in self._msg_events:
            counts[ev.msg_type] += 1
            totals[ev.msg_type] += ev.size_bytes
        return {
            t: {"count": counts[t], "bytes": totals[t]}
            for t in ("DIS", "DIO", "DAO")
        }

    def packet_delivery_ratio(self) -> float:
        if self.tx_attempts == 0:
            return 1.0
        delivered = self.tx_attempts - self.tx_lost
        return max(0.0, delivered / self.tx_attempts)

    def average_hop_count(self, nodes: dict) -> float:
        """Mean hop count across all associated non-root nodes."""
        hops = [
            n.hop_count_to_root()
            for n in nodes.values()
            if not n.is_root and n.state.name == "ASSOCIATED"
        ]
        return sum(hops) / len(hops) if hops else 0.0

    def fraction_joined(self, root_id: int = 0) -> float:
        non_root = [r for nid, r in self._nodes.items() if nid != root_id]
        if not non_root:
            return 1.0
        joined = sum(1 for r in non_root if r.join_time < float("inf"))
        return joined / len(non_root)

    def routes_installed_count(self) -> int:
        return len(self._routes)

    # ── Summary dict (used by sweep) ──────────────────────────────────────────

    def summary(self, nodes: dict) -> Dict:
        ovhd = self.message_overhead()
        total_nodes = len(self._nodes)
        root_id = next((nid for nid, n in nodes.items() if n.is_root), 0)
        joined = sum(
            1 for nid, r in self._nodes.items()
            if r.join_time < float("inf") or nid == root_id
        )
        return {
            "total_nodes": total_nodes,
            "joined_nodes": joined,
            "convergence_fraction": joined / total_nodes if total_nodes else 0.0,
            "convergence_time_s": self.convergence_time(root_id),
            "dis_count": ovhd["DIS"]["count"],
            "dio_count": ovhd["DIO"]["count"],
            "dao_count": ovhd["DAO"]["count"],
            "dis_bytes": ovhd["DIS"]["bytes"],
            "dio_bytes": ovhd["DIO"]["bytes"],
            "dao_bytes": ovhd["DAO"]["bytes"],
            "total_control_bytes": sum(ovhd[t]["bytes"] for t in ("DIS", "DIO", "DAO")),
            "pdr": self.packet_delivery_ratio(),
            "avg_hop_count": self.average_hop_count(nodes),
            "routes_installed": self.routes_installed_count(),
        }

    # ── CSV export ────────────────────────────────────────────────────────────

    def export_summary_csv(self, path: str, nodes: dict) -> None:
        row = self.summary(nodes)
        _write_csv(path, [row])
        logger.info(f"Summary written → {path}")

    def export_message_log_csv(self, path: str) -> None:
        rows = [
            {
                "time": ev.time,
                "sender_id": ev.sender_id,
                "msg_type": ev.msg_type,
                "size_bytes": ev.size_bytes,
            }
            for ev in self._msg_events
        ]
        _write_csv(path, rows)
        logger.info(f"Message log written → {path}")

    def export_node_log_csv(self, path: str, root_id: int = 0) -> None:
        rows = [
            {
                "node_id": r.node_id,
                "join_time": r.join_time if r.join_time < float("inf") else -1,
                "parent_id": r.parent_id if r.parent_id is not None else -1,
                "final_rank": r.final_rank,
                "dis_sent": r.sent["DIS"],
                "dio_sent": r.sent["DIO"],
                "dao_sent": r.sent["DAO"],
                "bytes_sent": r.bytes_sent,
            }
            for r in self._nodes.values()
        ]
        _write_csv(path, rows)
        logger.info(f"Node log written → {path}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
