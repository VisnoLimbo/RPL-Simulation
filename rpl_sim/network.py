"""Network topology and wireless radio medium simulation.

The Network class:
  - Owns all Node instances.
  - Models link reachability by Euclidean distance vs. radio range.
  - Applies per-link probabilistic packet loss (iid Bernoulli drops).
  - Routes broadcast and unicast messages through SimPy timeout events
    to simulate propagation delay.

Two convenience factory functions are provided:

  create_random_network()  – uniform random placement.
  create_grid_network()    – deterministic grid layout.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, Optional, Set, Tuple

import simpy

from .metrics import MetricsCollector
from .node import Node
from .rpl import ObjectiveFunction, OF0, create_objective_function

logger = logging.getLogger(__name__)

# One millisecond propagation delay (all links assumed equal)
PROPAGATION_DELAY: float = 0.001


class Network:
    """Wireless network hosting RPL nodes.

    Parameters
    ----------
    env:               SimPy environment.
    metrics:           Shared metrics collector.
    loss_probability:  Per-transmission packet loss probability [0, 1).
    rng:               Seeded random source for reproducibility.
    """

    def __init__(
        self,
        env: simpy.Environment,
        metrics: MetricsCollector,
        loss_probability: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.env = env
        self.metrics = metrics
        self.loss_probability = max(0.0, min(loss_probability, 0.99))
        self.rng = rng or random.Random()

        self.nodes: Dict[int, Node] = {}
        self._adj: Dict[int, Set[int]] = {}   # adjacency list (bidirectional)

    # ── Node management ───────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self._adj[node.node_id] = set()

    def build_topology(self) -> None:
        """Populate adjacency list from radio-range geometry.

        Must be called once after all nodes have been added.
        """
        node_list = list(self.nodes.values())
        link_count = 0
        for i, a in enumerate(node_list):
            for b in node_list[i + 1:]:
                d = _dist(a.position, b.position)
                if d <= a.radio_range and d <= b.radio_range:
                    self._adj[a.node_id].add(b.node_id)
                    self._adj[b.node_id].add(a.node_id)
                    link_count += 1
        logger.info(
            f"Topology: {len(self.nodes)} nodes, {link_count} bidirectional links"
        )

    def get_neighbors(self, node_id: int) -> Set[int]:
        return self._adj.get(node_id, set())

    def get_link_metric(self, from_id: int, to_id: int) -> float:
        """ETX estimate for a link.

        With zero loss the ETX is 1.0 (perfect link).
        With loss probability *p*, ETX = 1 / (1-p)^2 modelling a
        symmetric, independent bidirectional link.
        """
        if self.loss_probability <= 0:
            return 1.0
        prr = 1.0 - self.loss_probability
        return min(1.0 / (prr * prr), 10.0)   # cap at 10 to avoid rank overflow

    # ── Message delivery ──────────────────────────────────────────────────────

    def broadcast(self, sender_id: int, msg: object) -> None:
        """Deliver *msg* to every in-range neighbour, applying loss."""
        for nid in self._adj.get(sender_id, set()):
            self.metrics.record_tx_attempt()
            if self.rng.random() < self.loss_probability:
                self.metrics.record_packet_loss()
            else:
                self.env.process(self._deliver(nid, msg))

    def unicast(self, sender_id: int, target_id: int, msg: object) -> None:
        """Deliver *msg* to *target_id* if it is a direct neighbour."""
        if target_id not in self._adj.get(sender_id, set()):
            logger.warning(
                f"Unicast from {sender_id} to {target_id} failed: not a neighbour"
            )
            return
        self.metrics.record_tx_attempt()
        if self.rng.random() < self.loss_probability:
            self.metrics.record_packet_loss()
        else:
            self.env.process(self._deliver(target_id, msg))

    def _deliver(self, node_id: int, msg: object):
        yield self.env.timeout(PROPAGATION_DELAY)
        node = self.nodes.get(node_id)
        if node:
            node.receive(msg)

    # ── Graph utilities ───────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """True if every node is reachable from the root via radio links."""
        root_id = next((nid for nid, n in self.nodes.items() if n.is_root), None)
        if root_id is None or not self.nodes:
            return False
        visited: Set[int] = {root_id}
        queue = [root_id]
        while queue:
            cur = queue.pop()
            for nb in self._adj.get(cur, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == len(self.nodes)

    def dodag_is_valid(self) -> Tuple[bool, str]:
        """Check that the formed DODAG is loop-free and all nodes are joined."""
        from .rpl import NodeState, INFINITE_RANK
        for nid, node in self.nodes.items():
            if node.is_root:
                continue
            if node.state != NodeState.ASSOCIATED:
                return False, f"Node {nid} is not ASSOCIATED"
            # Walk path to root; check no loops
            path = node.get_path_to_root()
            if len(set(path)) != len(path):
                return False, f"Loop in path for node {nid}: {path}"
            # Parent rank must be lower
            parent = self.nodes.get(node.preferred_parent)
            if parent and parent.rank >= node.rank:
                return False, (
                    f"Rank inversion: node {nid} rank={node.rank}, "
                    f"parent {parent.node_id} rank={parent.rank}"
                )
        return True, "OK"

    def average_path_length(self) -> float:
        """Average hop distance from non-root nodes to root."""
        hops = [
            n.hop_count_to_root()
            for n in self.nodes.values()
            if not n.is_root and n.state.name == "ASSOCIATED"
        ]
        return sum(hops) / len(hops) if hops else 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── Factory functions ─────────────────────────────────────────────────────────

def create_random_network(
    env: simpy.Environment,
    metrics: MetricsCollector,
    num_nodes: int,
    area_size: float,
    radio_range: float,
    loss_probability: float = 0.0,
    objective_function: Optional[ObjectiveFunction] = None,
    use_trickle: bool = True,
    trickle_imin: float = 1.0,
    trickle_imax_doublings: int = 8,
    trickle_k: int = 10,
    seed: int = 42,
    ensure_connected: bool = True,
    max_retries: int = 20,
) -> Network:
    """Create a network with *num_nodes* nodes placed uniformly at random.

    The root (node 0) is placed at the centre of the area.
    If *ensure_connected* is True, placement is retried until the
    resulting radio graph is connected (up to *max_retries* attempts).

    Parameters
    ----------
    num_nodes:       Total nodes including 1 root.
    area_size:       Side length of the square deployment area (metres).
    radio_range:     Transmission range (metres).
    loss_probability:Per-link Bernoulli loss probability.
    objective_function: OF instance; defaults to OF0.
    seed:            Random seed for reproducibility.
    """
    of = objective_function or OF0()
    trickle_kw = dict(
        trickle_imin=trickle_imin,
        trickle_imax_doublings=trickle_imax_doublings,
        trickle_k=trickle_k,
    )

    for attempt in range(max_retries):
        rng = random.Random(seed + attempt)
        net = Network(env, metrics, loss_probability, rng)

        # Root at centre
        root = Node(
            env=env, node_id=0,
            position=(area_size / 2.0, area_size / 2.0),
            radio_range=radio_range, network=net, metrics=metrics,
            is_root=True, of=of, use_trickle=use_trickle,
            rng=random.Random(seed),
            **trickle_kw,
        )
        net.add_node(root)

        for i in range(1, num_nodes):
            pos = (rng.uniform(0, area_size), rng.uniform(0, area_size))
            node = Node(
                env=env, node_id=i,
                position=pos,
                radio_range=radio_range, network=net, metrics=metrics,
                is_root=False, of=of, use_trickle=use_trickle,
                rng=random.Random(seed + i * 1000 + attempt),
                **trickle_kw,
            )
            net.add_node(node)

        net.build_topology()

        if not ensure_connected or net.is_connected():
            if attempt > 0:
                logger.info(f"Connected topology found on attempt {attempt + 1}")
            return net

        logger.warning(
            f"Topology attempt {attempt + 1}/{max_retries} not connected, retrying..."
        )

    logger.error("Could not generate a connected topology; returning last attempt")
    return net  # type: ignore[return-value]


def create_grid_network(
    env: simpy.Environment,
    metrics: MetricsCollector,
    rows: int,
    cols: int,
    spacing: float,
    radio_range: float,
    loss_probability: float = 0.0,
    objective_function: Optional[ObjectiveFunction] = None,
    use_trickle: bool = True,
    trickle_imin: float = 1.0,
    trickle_imax_doublings: int = 8,
    trickle_k: int = 10,
    seed: int = 42,
) -> Network:
    """Create a *rows* × *cols* grid network.

    Node 0 (root) is placed at position (0, 0).  Other nodes are laid
    out left-to-right, top-to-bottom.  Useful for deterministic tests.
    """
    of = objective_function or OF0()
    trickle_kw = dict(
        trickle_imin=trickle_imin,
        trickle_imax_doublings=trickle_imax_doublings,
        trickle_k=trickle_k,
    )
    rng = random.Random(seed)
    net = Network(env, metrics, loss_probability, rng)

    node_id = 0
    for r in range(rows):
        for c in range(cols):
            pos = (c * spacing, r * spacing)
            is_root = (node_id == 0)
            node = Node(
                env=env, node_id=node_id,
                position=pos,
                radio_range=radio_range, network=net, metrics=metrics,
                is_root=is_root, of=of, use_trickle=use_trickle,
                rng=random.Random(seed + node_id),
                **trickle_kw,
            )
            net.add_node(node)
            node_id += 1

    net.build_topology()
    return net
