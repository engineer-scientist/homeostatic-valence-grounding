"""Decoupled interoception: displayed gauges do not track the agent's own body."""

from __future__ import annotations

from collections import deque

import numpy as np

from hvg.env.chargeworld import ChargeWorld


class GaugeYoke:
    """Produce displayed gauges from a delay, a shuffle buffer, or a logged stream."""

    def __init__(
        self,
        mode: str = "delay",
        delay: int = 20,
        buffer_size: int = 64,
        seed: int | None = None,
        log: list[tuple[float, float]] | None = None,
    ):
        if mode not in {"delay", "shuffle", "yoke_log"}:
            raise ValueError(f"unknown yoke mode: {mode}")
        self.mode = mode
        self.delay = int(delay)
        self.buffer_size = int(buffer_size)
        self.rng = np.random.default_rng(seed)
        self.buf: deque[tuple[float, float]] = deque(maxlen=max(buffer_size, delay + 1))
        self.log = list(log or [])
        self._i = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.buf.clear()
        self._i = 0

    def display(self, true_e: float, true_i: float) -> tuple[float, float]:
        if self.mode == "yoke_log":
            if not self.log:
                return true_e, true_i
            e, i = self.log[self._i % len(self.log)]
            self._i += 1
            return float(e), float(i)
        self.buf.append((true_e, true_i))
        if self.mode == "delay":
            if len(self.buf) <= self.delay:
                # Before the buffer fills, show a mid, uninformative pair
                # so the agent cannot read the current state "for free."
                return 0.5, 0.5
            e, i = self.buf[0] if self.delay > 0 else (true_e, true_i)
            # deque[0] is the oldest when maxlen isn't exceeded; with append
            # and maxlen, index 0 is oldest. For delay d we want the value
            # from d steps ago: that's buf[0] when len == d+1 and maxlen >= d+1.
            return float(e), float(i)
        # shuffle: sample a past (or current) pair, not biased to the latest
        idx = int(self.rng.integers(0, len(self.buf)))
        e, i = list(self.buf)[idx]
        return float(e), float(i)


class DecoupledChargeWorld(ChargeWorld):
    """Same physics as ChargeWorld; observations show yoked / shuffled gauges."""

    def __init__(self, physics: dict | None = None, seed: int | None = None, yoke: GaugeYoke | None = None):
        super().__init__(physics=physics, seed=seed)
        self.yoke = yoke or GaugeYoke()
        self._disp = (0.5, 0.5)

    def reset(self, seed: int | None = None, layout: dict | None = None):
        self.yoke.reset(seed=seed)
        obs, info = super().reset(seed=seed, layout=layout)
        # Recompute displayed gauges through the yoke.
        self._disp = self.yoke.display(self.energy, self.integrity)
        return self.observe(), self.info()

    def displayed_gauges(self) -> tuple[float, float]:
        return self._disp

    def step(self, action: int):
        obs, reward, term, trunc, info = super().step(action)
        self._disp = self.yoke.display(self.energy, self.integrity)
        # Refresh observation so the policy sees the yoked pair.
        obs = self.observe()
        info["displayed_energy"] = self._disp[0]
        info["displayed_integrity"] = self._disp[1]
        return obs, reward, term, trunc, info
