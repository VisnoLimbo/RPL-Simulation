"""Trickle algorithm (RFC 6206).

Controls the rate of DIO transmissions:
  - Start with a short interval Imin for fast convergence.
  - Double the interval each round (up to Imax) when the network is
    stable (few topology changes).
  - Suppress the transmission if enough neighbours have already
    broadcast a consistent message (counter c >= k).
  - Reset to Imin on detecting an inconsistency (topology change,
    new DODAG version, etc.).
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

import simpy

logger = logging.getLogger(__name__)


class TrickleTimer:
    """RFC 6206 Trickle timer wired into a SimPy environment.

    Parameters
    ----------
    env:
        SimPy simulation environment.
    callback:
        Zero-argument callable invoked when the timer fires at time *t*
        within the current interval (if suppression does not apply).
    imin:
        Minimum interval size in simulation time units.
    imax_doublings:
        Number of doublings before the interval is capped.
        Maximum interval = imin × 2^imax_doublings.
    k:
        Redundancy constant.  k=0 disables suppression (always transmit).
    rng:
        Optional seeded :class:`random.Random` for reproducibility.
    """

    def __init__(
        self,
        env: simpy.Environment,
        callback: Callable[[], None],
        imin: float = 1.0,
        imax_doublings: int = 8,
        k: int = 10,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.env = env
        self.callback = callback
        self.imin = imin
        self.imax = imin * (2 ** imax_doublings)
        self.k = k
        self.rng = rng or random.Random()

        self._I: float = imin         # Current interval size
        self._c: int = 0              # Consistency counter
        self._running: bool = False
        self._process: Optional[simpy.Process] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Trickle timer (idempotent)."""
        if not self._running:
            self._running = True
            self._I = self.imin
            self._c = 0
            self._process = self.env.process(self._run())

    def stop(self) -> None:
        """Stop the timer and cancel the underlying SimPy process."""
        self._running = False
        if self._process and self._process.is_alive:
            try:
                self._process.interrupt("stop")
            except RuntimeError:
                pass

    def hear_consistent(self) -> None:
        """Record a consistent message received from a neighbour.

        A consistent message is one that carries the same DODAG ID,
        version, and rank as the local node already knows.  Incrementing
        *c* may suppress the next transmission.
        """
        self._c += 1

    def hear_inconsistent(self) -> None:
        """Record an inconsistency (topology change, new version, etc.).

        If the current interval is already Imin there is nothing to
        collapse.  Otherwise reset to Imin to trigger rapid re-convergence.
        """
        if self._I > self.imin:
            self._I = self.imin
            self._c = 0
            if self._process and self._process.is_alive:
                try:
                    self._process.interrupt("reset")
                except RuntimeError:
                    pass

    # ── Internal ────────────────────────────────────────────────────────────

    def _run(self) -> simpy.events.Process:
        """Main Trickle loop (SimPy generator)."""
        while self._running:
            try:
                t = self.rng.uniform(self._I / 2.0, self._I)
                remaining = self._I - t

                yield self.env.timeout(t)

                if self.k == 0 or self._c < self.k:
                    self.callback()
                    logger.debug(
                        f"[t={self.env.now:.3f}] Trickle fired "
                        f"(I={self._I:.2f}, c={self._c}, k={self.k})"
                    )

                yield self.env.timeout(remaining)

                # Advance to next interval
                self._I = min(self._I * 2.0, self.imax)
                self._c = 0

            except simpy.Interrupt as intr:
                if intr.cause == "stop":
                    break
                # "reset" or any other cause: restart with updated self._I
                continue
