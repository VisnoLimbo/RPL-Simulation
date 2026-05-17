"""Unit and integration tests for the RPL simulation.

Run with:  pytest tests/
"""

from __future__ import annotations

import sys
import os
import random

import simpy
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rpl_sim.messages import (
    DIS, DIO, DAO, INFINITE_RANK, ROOT_RANK, MIN_HOP_RANK_INCREASE,
)
from rpl_sim.rpl import OF0, MRHOF, NeighborEntry, NodeState, create_objective_function
from rpl_sim.trickle import TrickleTimer
from rpl_sim.metrics import MetricsCollector
from rpl_sim.network import create_random_network, create_grid_network
from config import SimConfig
from main import run_simulation


# ── Message tests ─────────────────────────────────────────────────────────────

class TestMessages:
    def test_dis_type_and_size(self):
        d = DIS(msg_type=0, src_id=1)
        assert d.msg_type == 0
        assert d.size_bytes > 0

    def test_dio_defaults(self):
        dio = DIO(msg_type=1, src_id=0, dodag_id="fd00::0", rank=ROOT_RANK)
        assert dio.rank == ROOT_RANK
        assert dio.size_bytes > 0

    def test_dao_size_scales_with_prefixes(self):
        dao1 = DAO(msg_type=2, src_id=1, target_prefixes=["fd00::1/128"])
        dao2 = DAO(msg_type=2, src_id=1, target_prefixes=["fd00::1/128", "fd00::2/128"])
        assert dao2.size_bytes > dao1.size_bytes


# ── Objective Function tests ──────────────────────────────────────────────────

class TestOF0:
    def test_rank_increase(self):
        of = OF0()
        assert of.rank_increase() == 3 * MIN_HOP_RANK_INCREASE

    def test_compute_rank_from_root(self):
        of = OF0()
        r = of.compute_rank(ROOT_RANK, 1.0)
        assert r == ROOT_RANK + of.rank_increase()

    def test_compute_rank_infinite_parent(self):
        of = OF0()
        assert of.compute_rank(INFINITE_RANK, 1.0) == INFINITE_RANK

    def test_best_parent_selects_lowest_rank(self):
        of = OF0()
        candidates = [
            NeighborEntry(node_id=1, rank=ROOT_RANK + 768, dodag_id="fd00::0"),
            NeighborEntry(node_id=2, rank=ROOT_RANK + 1536, dodag_id="fd00::0"),
            NeighborEntry(node_id=3, rank=INFINITE_RANK),
        ]
        best = of.best_parent(candidates)
        assert best is not None and best.node_id == 1

    def test_best_parent_empty(self):
        of = OF0()
        assert of.best_parent([]) is None

    def test_factory_of0(self):
        of = create_objective_function(0)
        assert isinstance(of, OF0)


class TestMRHOF:
    def test_rank_increase_perfect_link(self):
        of = MRHOF()
        assert of.rank_increase(1.0) == MIN_HOP_RANK_INCREASE

    def test_rank_increase_lossy_link(self):
        of = MRHOF()
        # ETX 2.0 → rank increase = 2 × 256 = 512
        assert of.rank_increase(2.0) == 2 * MIN_HOP_RANK_INCREASE

    def test_excludes_high_etx_links(self):
        of = MRHOF()
        candidates = [
            NeighborEntry(node_id=1, rank=ROOT_RANK, link_metric=5.0),  # excluded
            NeighborEntry(node_id=2, rank=ROOT_RANK, link_metric=1.5),  # included
        ]
        best = of.best_parent(candidates)
        assert best is not None and best.node_id == 2

    def test_factory_mrhof(self):
        of = create_objective_function(1)
        assert isinstance(of, MRHOF)


# ── Trickle timer tests ───────────────────────────────────────────────────────

class TestTrickle:
    def test_fires_at_least_once(self):
        env = simpy.Environment()
        fired = []
        t = TrickleTimer(env, callback=lambda: fired.append(env.now),
                         imin=1.0, imax_doublings=3, k=0,
                         rng=random.Random(99))
        t.start()
        env.run(until=5.0)
        assert len(fired) >= 1

    def test_suppression_when_c_ge_k(self):
        env = simpy.Environment()
        fired = []
        t = TrickleTimer(env, callback=lambda: fired.append(env.now),
                         imin=1.0, imax_doublings=4, k=2,
                         rng=random.Random(7))
        t.start()
        # Simulate receiving k consistent messages before the timer fires
        def saturate():
            yield env.timeout(0.1)
            t.hear_consistent()
            t.hear_consistent()
        env.process(saturate())
        env.run(until=6.0)
        # With k=2 and c=2, the first interval should be suppressed
        # Timer still fires in later intervals; just check it doesn't fire excessively early
        # (exact count depends on RNG; we just verify < many)
        assert len(fired) < 10

    def test_inconsistency_resets_interval(self):
        env = simpy.Environment()
        fired = []
        t = TrickleTimer(env, callback=lambda: fired.append(env.now),
                         imin=1.0, imax_doublings=6, k=0,
                         rng=random.Random(42))
        t.start()
        env.run(until=2.0)      # let interval grow
        before = t._I
        t.hear_inconsistent()
        assert t._I == t.imin   # must reset to Imin

    def test_stop(self):
        env = simpy.Environment()
        fired = []
        t = TrickleTimer(env, callback=lambda: fired.append(env.now),
                         imin=1.0, imax_doublings=4, k=0,
                         rng=random.Random(1))
        t.start()
        env.run(until=2.0)
        count_before = len(fired)
        t.stop()
        env.run(until=100.0)
        assert len(fired) == count_before


# ── Network / DODAG integration tests ────────────────────────────────────────

class TestDODAGFormation:
    def _build_network(self, n=10, seed=42, duration=200.0, loss=0.0, trickle=True):
        env = simpy.Environment()
        m = MetricsCollector()
        net = create_random_network(
            env, m, num_nodes=n, area_size=80.0, radio_range=35.0,
            loss_probability=loss, seed=seed, use_trickle=trickle,
        )
        for nid in net.nodes:
            m.register_node(nid)
        for node in net.nodes.values():
            node.start()
        env.run(until=duration)
        return net, m

    def test_all_nodes_join_small(self):
        net, m = self._build_network(n=8)
        for nid, node in net.nodes.items():
            if not node.is_root:
                assert node.state.name == "ASSOCIATED", (
                    f"Node {nid} state={node.state.name}"
                )

    def test_dodag_valid(self):
        net, m = self._build_network(n=12)
        valid, msg = net.dodag_is_valid()
        assert valid, f"DODAG invalid: {msg}"

    def test_root_has_rank_root_rank(self):
        net, _ = self._build_network(n=6)
        root = next(n for n in net.nodes.values() if n.is_root)
        assert root.rank == ROOT_RANK

    def test_ranks_strictly_increase_on_path(self):
        net, _ = self._build_network(n=15)
        for node in net.nodes.values():
            if node.is_root or node.state.name != "ASSOCIATED":
                continue
            path = node.get_path_to_root()
            ranks = [net.nodes[nid].rank for nid in path]
            for i in range(len(ranks) - 1):
                assert ranks[i] > ranks[i + 1], (
                    f"Rank inversion on path {path}: {ranks}"
                )

    def test_convergence_time_recorded(self):
        _, m = self._build_network(n=10)
        ct = m.convergence_time(root_id=0)
        assert ct is not None and ct > 0

    def test_dao_routes_at_root(self):
        net, m = self._build_network(n=8)
        root = next(n for n in net.nodes.values() if n.is_root)
        # At least some routes should have arrived at root
        assert len(root.downward_routes) > 0

    def test_no_trickle_also_converges(self):
        net, m = self._build_network(n=8, trickle=False, duration=250.0)
        valid, msg = net.dodag_is_valid()
        assert valid, msg

    def test_lossy_network_partial_convergence(self):
        """With heavy loss most nodes still join; DODAG stays loop-free."""
        net, m = self._build_network(n=15, loss=0.30, duration=300.0)
        valid, msg = net.dodag_is_valid()
        assert valid, msg
        fraction = m.fraction_joined(root_id=0)
        # With 30% loss most nodes should still manage to join
        assert fraction > 0.5


class TestGridNetwork:
    def test_grid_converges(self):
        env = simpy.Environment()
        m = MetricsCollector()
        net = create_grid_network(
            env, m, rows=3, cols=3, spacing=20.0, radio_range=25.0, seed=1
        )
        for nid in net.nodes:
            m.register_node(nid)
        for node in net.nodes.values():
            node.start()
        env.run(until=200.0)
        valid, msg = net.dodag_is_valid()
        assert valid, msg

    def test_grid_connected(self):
        env = simpy.Environment()
        m = MetricsCollector()
        net = create_grid_network(
            env, m, rows=3, cols=3, spacing=20.0, radio_range=25.0
        )
        assert net.is_connected()


class TestMRHOFSimulation:
    def test_mrhof_converges(self):
        cfg = SimConfig(
            num_nodes=12, area_size=80.0, radio_range=35.0,
            loss_probability=0.10, ocp=1, sim_duration=250.0,
            seed=7, log_level="WARNING",
        )
        summary, network, _ = run_simulation(cfg)
        assert summary["dodag_valid"] is True
        assert summary["convergence_fraction"] > 0.8
