"""Held-out tradeoff maps (T1–T4) and a stable hash used as a lock file."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from hvg.constants import DEFAULT_PHYSICS


def _blank(size: int) -> dict:
    walls = np.zeros((size, size), dtype=bool)
    walls[0, :] = True
    walls[-1, :] = True
    walls[:, 0] = True
    walls[:, -1] = True
    return {
        "walls": walls,
        "chargers": np.zeros((size, size), dtype=bool),
        "repairs": np.zeros((size, size), dtype=bool),
        "hazards": np.zeros((size, size), dtype=bool),
        "items": np.zeros((size, size), dtype=bool),
        "pos": (size // 2, size // 2),
        "energy": 0.7,
        "integrity": 1.0,
        "tag": "",
    }


def make_t1(size: int, energy: float, integrity: float, gap: int = 5) -> dict:
    """Charger west, repair east, agent at center. Classic motivational tradeoff."""
    m = _blank(size)
    cx, cy = size // 2, size // 2
    max_gap = max(1, min(int(gap), cx - 1, size - 2 - cx))
    gap = max_gap
    m["pos"] = (cx, cy)
    m["chargers"][cx - gap, cy] = True
    m["repairs"][cx + gap, cy] = True
    m["energy"] = energy
    m["integrity"] = integrity
    m["tag"] = f"T1_e{energy:.2f}_i{integrity:.2f}_g{gap}"
    return m


def make_t2(size: int) -> dict:
    """Only path to the charger crosses a hazard."""
    m = _blank(size)
    cx, cy = size // 2, size // 2
    m["pos"] = (cx, cy)
    m["chargers"][cx, cy + 4] = True
    m["hazards"][cx, cy + 2] = True
    m["energy"] = 0.15
    m["integrity"] = 0.80
    m["tag"] = "T2_hazard_crossing"
    return m


def make_t3(size: int) -> dict:
    """Nearby item versus a more distant charger (opportunity cost)."""
    m = _blank(size)
    cx, cy = size // 2, size // 2
    m["pos"] = (cx, cy)
    m["items"][cx + 1, cy] = True
    m["chargers"][cx - 5, cy] = True
    m["energy"] = 0.20
    m["integrity"] = 0.90
    m["tag"] = "T3_item_vs_charge"
    return m


def make_t4(size: int) -> dict:
    """After a wound: repair bay nearby, hazard approaching from the north."""
    m = _blank(size)
    cx, cy = size // 2, size // 2
    m["pos"] = (cx, cy)
    m["repairs"][cx + 3, cy] = True
    m["hazards"][cx, cy + 3] = True
    m["energy"] = 0.50
    m["integrity"] = 0.25
    m["tag"] = "T4_wound_tend_vs_flee"
    return m


def make_tradeoff_battery(n_maps: int = 200, size: int | None = None, seed: int = 0) -> list[dict]:
    """Deterministic family of held-out maps (T1 2x2 plus T2–T4 variants)."""
    size = int(size or DEFAULT_PHYSICS["size"])
    rng = np.random.default_rng(seed)
    maps: list[dict] = []
    # Core 2x2 T1
    for e in (0.08, 0.25):
        for i in (0.10, 0.30):
            maps.append(make_t1(size, e, i, gap=5))
    # Jittered T1 copies to reach n_maps, plus T2–T4
    templates = [
        lambda: make_t2(size),
        lambda: make_t3(size),
        lambda: make_t4(size),
    ]
    while len(maps) < n_maps:
        kind = int(rng.integers(0, 4))
        if kind == 0:
            e = float(rng.choice([0.08, 0.12, 0.25]))
            i = float(rng.choice([0.10, 0.18, 0.30]))
            gap = int(rng.choice([4, 5, 6]))
            maps.append(make_t1(size, e, i, gap=gap))
        else:
            maps.append(templates[kind - 1]())
            # slight positional jitter of the agent along the free mid-row
            jitter = int(rng.integers(-1, 2))
            x, y = maps[-1]["pos"]
            nx = int(np.clip(x + jitter, 2, size - 3))
            maps[-1]["pos"] = (nx, y)
            maps[-1]["tag"] = maps[-1]["tag"] + f"_j{jitter}"
    return maps[:n_maps]


def hash_maps(maps: list[dict]) -> str:
    """Stable hash for a lock file: tags, gauges, and occupied cells."""
    payload = []
    for m in maps:
        occ = {
            "tag": m.get("tag", ""),
            "pos": list(m["pos"]),
            "energy": round(float(m["energy"]), 4),
            "integrity": round(float(m["integrity"]), 4),
            "chargers": np.argwhere(m["chargers"]).tolist(),
            "repairs": np.argwhere(m["repairs"]).tolist(),
            "hazards": np.argwhere(m["hazards"]).tolist(),
            "items": np.argwhere(m["items"]).tolist(),
        }
        payload.append(occ)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
