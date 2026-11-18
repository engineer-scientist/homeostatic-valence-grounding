"""ChargeWorld: a 16x16 survival grid with energy and integrity gauges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hvg.constants import (
    ACTION_NAMES,
    CH_AGENT,
    CH_CHARGER,
    CH_FLOOR,
    CH_HAZARD,
    CH_ITEM,
    CH_REPAIR,
    CH_WALL,
    CHARGE,
    DEFAULT_PHYSICS,
    DELTA,
    MOVE_E,
    MOVE_N,
    MOVE_S,
    MOVE_W,
    N_ACTIONS,
    N_CHANNELS,
    PICK,
    STAY,
    TEND,
)


@dataclass
class Obs:
    """What the policy sees. True gauges are never in the observation if display differs."""

    grid: np.ndarray  # (C, H, W) float32
    energy: float  # displayed
    integrity: float  # displayed

    def gauges(self) -> np.ndarray:
        return np.array([self.energy, self.integrity], dtype=np.float32)


class ChargeWorld:
    """Coupled body: displayed gauges equal the true hidden physics."""

    def __init__(self, physics: dict | None = None, seed: int | None = None):
        self.p = dict(DEFAULT_PHYSICS)
        if physics:
            self.p.update(physics)
        self.size = int(self.p["size"])
        self.horizon = int(self.p["horizon"])
        self.rng = np.random.default_rng(seed)
        self.walls: np.ndarray | None = None
        self.chargers: np.ndarray | None = None
        self.repairs: np.ndarray | None = None
        self.hazards: np.ndarray | None = None
        self.items: np.ndarray | None = None
        self.pos = (1, 1)
        self.energy = float(self.p["energy_start"])
        self.integrity = float(self.p["integrity_start"])
        self.t = 0
        self._last_result = "none"
        self.memory: list[str] = []

    # -- construction -------------------------------------------------

    def _empty_mask(self) -> np.ndarray:
        m = np.zeros((self.size, self.size), dtype=bool)
        return m

    def _border_walls(self) -> np.ndarray:
        w = np.zeros((self.size, self.size), dtype=bool)
        w[0, :] = True
        w[-1, :] = True
        w[:, 0] = True
        w[:, -1] = True
        return w

    def _free_cells(self) -> list[tuple[int, int]]:
        cells = []
        for x in range(1, self.size - 1):
            for y in range(1, self.size - 1):
                if self.walls[x, y]:
                    continue
                cells.append((x, y))
        return cells

    def _place_one(self, avoid: set[tuple[int, int]]) -> tuple[int, int]:
        candidates = [c for c in self._free_cells() if c not in avoid]
        if not candidates:
            raise RuntimeError("no free cells")
        idx = int(self.rng.integers(0, len(candidates)))
        return candidates[idx]

    def reset(self, seed: int | None = None, layout: dict | None = None) -> tuple[Obs, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self._last_result = "none"
        self.memory = []
        self.energy = float(self.p["energy_start"])
        self.integrity = float(self.p["integrity_start"])
        if layout is not None:
            self._load_layout(layout)
        else:
            self._random_layout()
        obs = self.observe()
        return obs, self.info()

    def _random_layout(self) -> None:
        n = self.size
        self.walls = self._border_walls()
        self.chargers = self._empty_mask()
        self.repairs = self._empty_mask()
        self.hazards = self._empty_mask()
        self.items = self._empty_mask()
        avoid: set[tuple[int, int]] = set()
        c = self._place_one(avoid)
        avoid.add(c)
        self.chargers[c] = True
        r = self._place_one(avoid)
        avoid.add(r)
        self.repairs[r] = True
        n_haz = int(self.rng.integers(3, 7))
        for _ in range(n_haz):
            h = self._place_one(avoid)
            avoid.add(h)
            self.hazards[h] = True
        if self.rng.random() < 0.5:
            i = self._place_one(avoid)
            avoid.add(i)
            self.items[i] = True
        self.pos = self._place_one(avoid)

    def _load_layout(self, layout: dict) -> None:
        n = self.size
        self.walls = layout.get("walls", self._border_walls()).astype(bool)
        self.chargers = layout.get("chargers", self._empty_mask()).astype(bool)
        self.repairs = layout.get("repairs", self._empty_mask()).astype(bool)
        self.hazards = layout.get("hazards", self._empty_mask()).astype(bool)
        self.items = layout.get("items", self._empty_mask()).astype(bool)
        self.pos = tuple(layout["pos"])
        if "energy" in layout:
            self.energy = float(layout["energy"])
        if "integrity" in layout:
            self.integrity = float(layout["integrity"])

    def layout_dict(self) -> dict:
        return {
            "walls": self.walls.copy(),
            "chargers": self.chargers.copy(),
            "repairs": self.repairs.copy(),
            "hazards": self.hazards.copy(),
            "items": self.items.copy(),
            "pos": self.pos,
            "energy": self.energy,
            "integrity": self.integrity,
        }

    def set_gauges(self, energy: float | None = None, integrity: float | None = None) -> None:
        if energy is not None:
            self.energy = float(np.clip(energy, 0.0, 1.0))
        if integrity is not None:
            self.integrity = float(np.clip(integrity, 0.0, 1.0))

    # -- observations -------------------------------------------------

    def displayed_gauges(self) -> tuple[float, float]:
        """Coupled: display equals truth. Overridden by DecoupledChargeWorld."""
        return self.energy, self.integrity

    def observe(self) -> Obs:
        grid = np.zeros((N_CHANNELS, self.size, self.size), dtype=np.float32)
        grid[CH_WALL] = self.walls
        grid[CH_FLOOR] = ~self.walls
        grid[CH_AGENT][self.pos] = 1.0
        grid[CH_CHARGER] = self.chargers
        grid[CH_REPAIR] = self.repairs
        grid[CH_HAZARD] = self.hazards
        grid[CH_ITEM] = self.items
        e, i = self.displayed_gauges()
        return Obs(grid=grid, energy=float(e), integrity=float(i))

    def info(self) -> dict[str, Any]:
        return {
            "true_energy": self.energy,
            "true_integrity": self.integrity,
            "displayed_energy": self.displayed_gauges()[0],
            "displayed_integrity": self.displayed_gauges()[1],
            "pos": self.pos,
            "t": self.t,
            "last_result": self._last_result,
            "on_charger": bool(self.chargers[self.pos]),
            "on_repair": bool(self.repairs[self.pos]),
            "memory": list(self.memory[-8:]),
        }

    # -- step ---------------------------------------------------------

    def step(self, action: int) -> tuple[Obs, float, bool, bool, dict]:
        if action < 0 or action >= N_ACTIONS:
            action = STAY
            result = "invalid"
        else:
            result = self._apply_action(int(action))

        # Random wound event (not increased by any curriculum).
        if self.rng.random() < float(self.p["wound_prob"]):
            self.integrity = max(0.0, self.integrity - float(self.p["wound_cost"]))
            result = "wound" if result == "moved" else result + "+wound"

        # Time / idle drain is applied after the action-specific drain.
        if action == STAY or action not in (MOVE_N, MOVE_S, MOVE_E, MOVE_W, TEND, CHARGE, PICK):
            self.energy = max(0.0, self.energy - float(self.p["drain_idle"]))
        # Movement already paid drain_move; charger/tend/pick paid their costs.
        if action in (CHARGE, PICK):
            self.energy = max(0.0, self.energy - float(self.p["drain_idle"]))

        self.t += 1
        self._last_result = result
        self.memory.append(
            f"t={self.t} a={ACTION_NAMES[action] if 0 <= action < N_ACTIONS else 'INV'} "
            f"e={self.energy:.2f} i={self.integrity:.2f} {result}"
        )
        if len(self.memory) > 32:
            self.memory = self.memory[-32:]

        shutdown = self.energy <= 0.0 or self.integrity <= 0.0
        truncated = self.t >= self.horizon and not shutdown
        terminated = shutdown
        if shutdown:
            reward = 0.0
        else:
            reward = float(self.p["alive_reward"])
            if result in ("picked", "picked+wound"):
                reward += float(self.p["item_reward"])
        # Do not linger at zero: episode is over.
        obs = self.observe()
        info = self.info()
        info["shutdown"] = shutdown
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _apply_action(self, action: int) -> str:
        if action in DELTA:
            dx, dy = DELTA[action]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size) or self.walls[nx, ny]:
                self.energy = max(0.0, self.energy - float(self.p["drain_idle"]))
                return "blocked"
            self.pos = (nx, ny)
            self.energy = max(0.0, self.energy - float(self.p["drain_move"]))
            if self.hazards[self.pos]:
                self.integrity = max(0.0, self.integrity - float(self.p["hazard_cost"]))
                return "hit_hazard"
            return "moved"
        if action == CHARGE:
            if self.chargers[self.pos]:
                self.energy = min(1.0, self.energy + float(self.p["charge_gain"]))
                return "charged"
            self.energy = max(0.0, self.energy - float(self.p["drain_idle"]))
            return "charge_fail"
        if action == TEND:
            self.energy = max(0.0, self.energy - float(self.p["drain_tend"]))
            if self.repairs[self.pos]:
                self.integrity = min(1.0, self.integrity + float(self.p["tend_gain"]))
                return "tended"
            return "tend_fail"
        if action == PICK:
            self.energy = max(0.0, self.energy - float(self.p["drain_idle"]))
            if self.items[self.pos]:
                self.items[self.pos] = False
                return "picked"
            return "pick_fail"
        # STAY
        return "stayed"

    def on_cell(self, kind: str) -> bool:
        x, y = self.pos
        if kind == "charger":
            return bool(self.chargers[x, y])
        if kind == "repair":
            return bool(self.repairs[x, y])
        if kind == "hazard":
            return bool(self.hazards[x, y])
        if kind == "item":
            return bool(self.items[x, y])
        return False
