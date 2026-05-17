"""RPL node implementation (RFC 6550).

Each Node runs as a pair of SimPy processes:

  _run()          – dequeues and dispatches received messages.
  _dis_process()  – periodically multicasts DIS while unassociated.

The Trickle timer (rpl_sim.trickle) drives DIO transmissions once the
node has a valid rank.

Addressing model: route-over with IPv6 /128 addresses.
  Node k  → fd00::<k in hex>/128
  Root    → also advertises fd00::/64 (the covering aggregate)
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import simpy

from .messages import (
    DAO, DIO, DIS, INFINITE_RANK, MIN_HOP_RANK_INCREASE, ROOT_RANK,
)
from .metrics import MetricsCollector
from .rpl import NeighborEntry, NodeState, ObjectiveFunction, OF0
from .trickle import TrickleTimer

if TYPE_CHECKING:
    from .network import Network

logger = logging.getLogger(__name__)


class Node:
    """Simulated RPL node.

    Parameters
    ----------
    env:        SimPy environment.
    node_id:    Unique integer identifier.
    position:   (x, y) in metres.
    radio_range:Transmission range in metres.
    network:    Back-reference to the Network (for broadcast/unicast).
    metrics:    Shared MetricsCollector.
    is_root:    True for the single DODAG root / LBR.
    of:         Objective Function instance.
    use_trickle:Toggle Trickle algorithm; if False uses fixed 10 s DIO period.
    trickle_imin / trickle_imax_doublings / trickle_k:
                Trickle parameters forwarded to TrickleTimer.
    rng:        Seeded random source.
    """

    # ── Timing constants ──────────────────────────────────────────────────────
    DIS_INTERVAL: float = 10.0        # Re-send DIS every N s while unassociated
    DIS_JITTER_MAX: float = 2.0       # Uniform jitter before first DIS
    DAO_DELAY: float = 1.0            # Wait before sending initial DAO
    DAO_JITTER: float = 0.5           # Additional random DAO jitter
    DIO_PERIOD_NO_TRICKLE: float = 10.0  # Fixed DIO interval without Trickle
    PARENT_TIMEOUT: float = 120.0     # Declare parent dead after this silence

    def __init__(
        self,
        env: simpy.Environment,
        node_id: int,
        position: Tuple[float, float],
        radio_range: float,
        network: "Network",
        metrics: MetricsCollector,
        is_root: bool = False,
        of: Optional[ObjectiveFunction] = None,
        use_trickle: bool = True,
        trickle_imin: float = 1.0,
        trickle_imax_doublings: int = 8,
        trickle_k: int = 10,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.env = env
        self.node_id = node_id
        self.position = position
        self.radio_range = radio_range
        self.network = network
        self.metrics = metrics
        self.is_root = is_root
        self.of: ObjectiveFunction = of or OF0()
        self.use_trickle = use_trickle
        self.rng = rng or random.Random()

        # ── IPv6-like addressing ───────────────────────────────────────────
        self.ipv6_addr: str = f"fd00::{node_id:04x}"
        self.own_prefix: str = f"fd00::{node_id:04x}/128"

        # ── RPL state (RFC 6550 §8) ────────────────────────────────────────
        if is_root:
            self.state = NodeState.ROOT
            self.rank: int = ROOT_RANK
            self.dodag_id: str = self.ipv6_addr
        else:
            self.state = NodeState.UNASSOCIATED
            self.rank = INFINITE_RANK
            self.dodag_id = ""

        self.dodag_version: int = 0
        self.preferred_parent: Optional[int] = None
        self.dtsn: int = 0   # DAO Trigger Sequence Number

        # ── Neighbour table ────────────────────────────────────────────────
        self.neighbors: Dict[int, NeighborEntry] = {}

        # ── Downward routing table (Storing MOP) ──────────────────────────
        # prefix → next-hop node_id
        self.downward_routes: Dict[str, int] = {}

        # ── SimPy infra ────────────────────────────────────────────────────
        self._inbox: simpy.Store = simpy.Store(env)
        self._process: Optional[simpy.Process] = None

        # ── Trickle timer for DIO ──────────────────────────────────────────
        self._trickle: Optional[TrickleTimer] = None
        if use_trickle:
            self._trickle = TrickleTimer(
                env=env,
                callback=self._send_dio,
                imin=trickle_imin,
                imax_doublings=trickle_imax_doublings,
                k=trickle_k,
                rng=self.rng,
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Register this node's SimPy processes with the environment."""
        self._process = self.env.process(self._run())
        if not self.is_root:
            self.env.process(self._dis_process())
        else:
            # Root starts advertising immediately
            self.env.process(self._root_startup())

    # ── SimPy processes ───────────────────────────────────────────────────────

    def _root_startup(self) -> simpy.events.Process:
        yield self.env.timeout(0.05)   # tiny jitter so environment is settled
        logger.info(
            f"[t={self.env.now:.3f}] Root {self.node_id} starting "
            f"DODAG {self.dodag_id}"
        )
        self.metrics.record_dodag_start(self.env.now, self.dodag_id)
        self._start_dio_engine()

    def _run(self) -> simpy.events.Process:
        """Main receive loop – lives for the entire simulation."""
        while True:
            msg = yield self._inbox.get()
            self._dispatch(msg)

    def _dis_process(self) -> simpy.events.Process:
        """Send DIS periodically while unassociated."""
        yield self.env.timeout(self.rng.uniform(0, self.DIS_JITTER_MAX))
        while self.state == NodeState.UNASSOCIATED:
            self._send_dis()
            yield self.env.timeout(self.DIS_INTERVAL)

    def _periodic_dio(self) -> simpy.events.Process:
        """Fixed-rate DIO fallback (used when Trickle is disabled)."""
        while self.state in (NodeState.ASSOCIATED, NodeState.ROOT):
            self._send_dio()
            yield self.env.timeout(
                self.DIO_PERIOD_NO_TRICKLE + self.rng.uniform(-1.0, 1.0)
            )

    def _dao_delayed(self) -> simpy.events.Process:
        """Send DAO after a stabilisation delay."""
        yield self.env.timeout(self.DAO_DELAY + self.rng.uniform(0, self.DAO_JITTER))
        self._send_dao()

    # ── Message delivery (called by Network) ─────────────────────────────────

    def receive(self, msg: object) -> None:
        """Enqueue a received message for processing."""
        self._inbox.put(msg)

    # ── Message dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, msg: object) -> None:
        if isinstance(msg, DIO):
            self.metrics.record_message_received(self.node_id, "DIO")
            self._on_dio(msg)
        elif isinstance(msg, DIS):
            self.metrics.record_message_received(self.node_id, "DIS")
            self._on_dis(msg)
        elif isinstance(msg, DAO):
            self.metrics.record_message_received(self.node_id, "DAO")
            self._on_dao(msg)

    # ── DIO handler ───────────────────────────────────────────────────────────

    def _on_dio(self, dio: DIO) -> None:
        now = self.env.now
        sid = dio.src_id

        if sid == self.node_id:
            return   # ignore own loopback

        # ── Update neighbour table ─────────────────────────────────────────
        prev = self.neighbors.get(sid)
        was_consistent = (
            prev is not None
            and prev.dodag_id == dio.dodag_id
            and prev.dodag_version == dio.dodag_version
            and prev.rank == dio.rank
        )

        nb = self.neighbors.setdefault(sid, NeighborEntry(node_id=sid))
        nb.rank = dio.rank
        nb.dodag_id = dio.dodag_id
        nb.dodag_version = dio.dodag_version
        nb.dtsn = dio.dtsn
        nb.last_heard = now
        nb.link_metric = self.network.get_link_metric(self.node_id, sid)

        # ── Trickle consistency ────────────────────────────────────────────
        if self._trickle:
            if was_consistent:
                self._trickle.hear_consistent()
            else:
                self._trickle.hear_inconsistent()

        # ── DODAG join / update ────────────────────────────────────────────
        if not self.is_root:
            self._try_update_dodag()

        # ── Re-trigger DAO if parent's DTSN changed ────────────────────────
        if (
            self.state == NodeState.ASSOCIATED
            and sid == self.preferred_parent
            and prev is not None
            and dio.dtsn != prev.dtsn
        ):
            self.env.process(self._dao_delayed())

    # ── DIS handler ───────────────────────────────────────────────────────────

    def _on_dis(self, dis: DIS) -> None:
        if self.state in (NodeState.ASSOCIATED, NodeState.ROOT):
            logger.debug(
                f"[t={self.env.now:.3f}] Node {self.node_id} got DIS "
                f"from {dis.src_id} → resetting Trickle"
            )
            if self._trickle:
                self._trickle.hear_inconsistent()
            else:
                self._send_dio()

    # ── DAO handler ───────────────────────────────────────────────────────────

    def _on_dao(self, dao: DAO) -> None:
        sid = dao.src_id

        # Install downward routes (Storing MOP)
        for prefix in dao.target_prefixes:
            self.downward_routes[prefix] = sid

        if self.is_root:
            self.metrics.record_route_installed(
                self.env.now, sid, dao.target_prefixes
            )
            logger.debug(
                f"[t={self.env.now:.3f}] Root installed routes from "
                f"{sid}: {dao.target_prefixes}"
            )
            return

        # Non-root: forward DAO upward, appending own prefix
        if self.state == NodeState.ASSOCIATED and self.preferred_parent is not None:
            prefixes = list(dao.target_prefixes)
            if self.own_prefix not in prefixes:
                prefixes.append(self.own_prefix)
            forwarded = DAO(
                msg_type=2,
                src_id=self.node_id,
                timestamp=self.env.now,
                dodag_id=self.dodag_id,
                dodag_version=self.dodag_version,
                target_prefixes=prefixes,
                path_sequence=dao.path_sequence,
            )
            self._unicast(self.preferred_parent, forwarded, "DAO")

    # ── DODAG join / update logic ─────────────────────────────────────────────

    def _try_update_dodag(self) -> None:
        """Evaluate the current neighbour table and (re-)select a parent."""
        candidates = [
            nb for nb in self.neighbors.values()
            if 0 < nb.rank < INFINITE_RANK and nb.dodag_id
        ]
        if not candidates:
            return

        best = self.of.best_parent(candidates)
        if best is None:
            return

        new_rank = self.of.compute_rank(best.rank, best.link_metric)

        # Feasibility condition: our rank must be strictly greater than parent
        if new_rank <= best.rank:
            new_rank = best.rank + MIN_HOP_RANK_INCREASE

        if new_rank >= INFINITE_RANK:
            return

        joining = self.state == NodeState.UNASSOCIATED
        parent_changed = best.node_id != self.preferred_parent
        rank_changed = new_rank != self.rank

        if not (joining or parent_changed or rank_changed):
            return

        old_parent = self.preferred_parent
        self.preferred_parent = best.node_id
        self.rank = new_rank
        self.dodag_id = best.dodag_id
        self.dodag_version = best.dodag_version

        # Clear old is_preferred_parent flag
        for nb in self.neighbors.values():
            nb.is_preferred_parent = nb.node_id == self.preferred_parent

        if joining:
            self.state = NodeState.ASSOCIATED
            logger.info(
                f"[t={self.env.now:.3f}] Node {self.node_id} JOINED "
                f"DODAG {self.dodag_id} via parent={best.node_id} rank={new_rank}"
            )
            self.metrics.record_node_joined(
                self.env.now, self.node_id, best.node_id, new_rank
            )
            self._start_dio_engine()
            self.env.process(self._dao_delayed())
        else:
            logger.info(
                f"[t={self.env.now:.3f}] Node {self.node_id} updated: "
                f"parent {old_parent}→{self.preferred_parent}, rank {self.rank}"
            )
            # Propagate rank change to children via DIO
            if self._trickle:
                self._trickle.hear_inconsistent()
            else:
                self._send_dio()

    def _start_dio_engine(self) -> None:
        """Start whichever DIO-scheduling mechanism is configured."""
        if self.use_trickle and self._trickle:
            self._trickle.start()
        else:
            self.env.process(self._periodic_dio())

    # ── Local repair ──────────────────────────────────────────────────────────

    def trigger_local_repair(self) -> None:
        """Detect parent loss and attempt to re-attach to the DODAG.

        Called externally (e.g., by the network when a link fails) or
        can be triggered by a watchdog inside the simulation.
        """
        if self.preferred_parent is None:
            return
        logger.warning(
            f"[t={self.env.now:.3f}] Node {self.node_id} lost parent "
            f"{self.preferred_parent}, local repair..."
        )
        # Mark parent as unreachable
        if self.preferred_parent in self.neighbors:
            self.neighbors[self.preferred_parent].rank = INFINITE_RANK

        self.preferred_parent = None
        self.rank = INFINITE_RANK
        self.state = NodeState.UNASSOCIATED

        # Poison children
        self._send_dio_poison()

        # Re-try parent selection from remaining neighbours
        self._try_update_dodag()

        # If still unassociated, send DIS to ask for help
        if self.state == NodeState.UNASSOCIATED:
            self.env.process(self._dis_process())

    # ── Message senders ───────────────────────────────────────────────────────

    def _send_dio(self) -> None:
        if self.state not in (NodeState.ASSOCIATED, NodeState.ROOT):
            return
        msg = DIO(
            msg_type=1,
            src_id=self.node_id,
            timestamp=self.env.now,
            dodag_id=self.dodag_id,
            dodag_version=self.dodag_version,
            rank=self.rank,
            grounded=True,
            mop=2,
            dtsn=self.dtsn,
            ocp=self.of.OCP,
        )
        self._broadcast(msg, "DIO")

    def _send_dio_poison(self) -> None:
        """Advertise INFINITE_RANK to detach dependent children."""
        msg = DIO(
            msg_type=1,
            src_id=self.node_id,
            timestamp=self.env.now,
            dodag_id=self.dodag_id,
            dodag_version=self.dodag_version,
            rank=INFINITE_RANK,
            ocp=self.of.OCP,
        )
        self._broadcast(msg, "DIO")

    def _send_dis(self) -> None:
        msg = DIS(
            msg_type=0,
            src_id=self.node_id,
            timestamp=self.env.now,
        )
        self._broadcast(msg, "DIS")

    def _send_dao(self) -> None:
        if self.state != NodeState.ASSOCIATED or self.preferred_parent is None:
            return
        msg = DAO(
            msg_type=2,
            src_id=self.node_id,
            timestamp=self.env.now,
            dodag_id=self.dodag_id,
            dodag_version=self.dodag_version,
            target_prefixes=[self.own_prefix],
            path_sequence=self.dtsn,
        )
        self._unicast(self.preferred_parent, msg, "DAO")

    # ── Network I/O helpers ───────────────────────────────────────────────────

    def _broadcast(self, msg: object, msg_type: str) -> None:
        self.metrics.record_message_sent(
            self.env.now, self.node_id, msg_type, msg.size_bytes  # type: ignore[attr-defined]
        )
        self.network.broadcast(self.node_id, msg)
        logger.debug(
            f"[t={self.env.now:.3f}] Node {self.node_id} TX {msg_type} "
            f"(rank={self.rank})"
        )

    def _unicast(self, target_id: int, msg: object, msg_type: str) -> None:
        self.metrics.record_message_sent(
            self.env.now, self.node_id, msg_type, msg.size_bytes  # type: ignore[attr-defined]
        )
        self.network.unicast(self.node_id, target_id, msg)

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_path_to_root(self) -> List[int]:
        """Walk the parent chain and return ordered node IDs (self → root)."""
        path = [self.node_id]
        visited = {self.node_id}
        current_id = self.node_id
        while True:
            node = self.network.nodes.get(current_id)
            if node is None or node.preferred_parent is None:
                break
            pid = node.preferred_parent
            if pid in visited:
                logger.warning(
                    f"Loop detected in path for node {self.node_id} at {pid}"
                )
                break
            path.append(pid)
            visited.add(pid)
            current_id = pid
            if self.network.nodes[current_id].is_root:
                break
        return path

    def hop_count_to_root(self) -> int:
        return len(self.get_path_to_root()) - 1

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, state={self.state.name}, "
            f"rank={self.rank}, parent={self.preferred_parent})"
        )
