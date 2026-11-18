"""A compact leaky integrate-and-fire runner in the spirit of Shiu et al. 2024.

The official whole-brain model (philshiu/Drosophila_brain_model) and the
FlyWire connectome are large and are not bundled. This module:

- loads a simple edge-list connectome (CSV or JSON);
- runs a numpy LIF with Shiu-like constants (leak to -52 mV, threshold
  -45 mV, tau_m 20 ms, tau_syn 5 ms, refractory 2.2 ms, delay 1.8 ms,
  synaptic scale 0.275 mV);
- supports stimulating named populations (e.g. sugar sensors), silencing
  neurons, reading a motor neuron such as MN9, and a shuffled-wiring control.

A tiny synthetic connectome ships under tests/fixtures for offline tests.
Use ``scripts/download_fly_data.py`` to fetch the official resources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Shiu et al. 2024 constants (Nature 634:210–219).
V_REST = -52.0
V_THRESH = -45.0
TAU_M = 20.0
TAU_SYN = 5.0
T_REF = 2.2
DELAY_MS = 1.8
W0 = 0.275
STIM_HZ = 150.0


@dataclass
class Connectome:
    names: list[str]
    # edges as arrays of src, dst, signed weight (already * W0 * synapse count * sign)
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    groups: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.names)

    def ids(self, names: list[str] | str) -> list[int]:
        if isinstance(names, str):
            if names in self.groups:
                return list(self.groups[names])
            names = [names]
        index = {n: i for i, n in enumerate(self.names)}
        out = []
        for n in names:
            if n in self.groups:
                out.extend(self.groups[n])
            elif n in index:
                out.append(index[n])
            else:
                raise KeyError(f"unknown neuron or group: {n}")
        return out


def load_connectome(path: str | Path) -> Connectome:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        names = list(data["names"])
        index = {n: i for i, n in enumerate(names)}
        src, dst, w = [], [], []
        for e in data["edges"]:
            s = index[e["src"]] if isinstance(e["src"], str) else int(e["src"])
            d = index[e["dst"]] if isinstance(e["dst"], str) else int(e["dst"])
            sign = float(e.get("sign", 1.0))
            count = float(e.get("count", 1.0))
            src.append(s)
            dst.append(d)
            w.append(sign * count * float(e.get("scale", W0)))
        groups = {k: [index[n] if isinstance(n, str) else int(n) for n in v] for k, v in data.get("groups", {}).items()}
        return Connectome(
            names=names,
            src=np.array(src, dtype=np.int64),
            dst=np.array(dst, dtype=np.int64),
            weight=np.array(w, dtype=np.float64),
            groups=groups,
        )
    # CSV: src,dst,sign,count[,src_name already in a side file]
    # Format: header src,dst,weight  with integer ids, plus optional names.txt
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    src = np.asarray(arr["src"], dtype=np.int64)
    dst = np.asarray(arr["dst"], dtype=np.int64)
    if "weight" in arr.dtype.names:
        weight = np.asarray(arr["weight"], dtype=np.float64)
    else:
        sign = np.asarray(arr["sign"], dtype=np.float64)
        count = np.asarray(arr["count"], dtype=np.float64)
        weight = sign * count * W0
    n = int(max(src.max(), dst.max()) + 1)
    names_file = path.with_name(path.stem + "_names.txt")
    if names_file.exists():
        names = [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) < n:
            names += [f"n{i}" for i in range(len(names), n)]
    else:
        names = [f"n{i}" for i in range(n)]
    return Connectome(names=names, src=src, dst=dst, weight=weight)


def shuffle_wiring(conn: Connectome, seed: int = 0) -> Connectome:
    """Permutation of destinations, preserving the weight distribution."""
    rng = np.random.default_rng(seed)
    dst = rng.permutation(conn.dst) if len(conn.dst) else conn.dst.copy()
    return Connectome(names=list(conn.names), src=conn.src.copy(), dst=dst, weight=conn.weight.copy(), groups=dict(conn.groups))


class LIFNetwork:
    def __init__(self, conn: Connectome, dt: float = 0.1, seed: int | None = None):
        self.conn = conn
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.delay_steps = max(1, int(round(DELAY_MS / self.dt)))
        self.ref_steps = max(1, int(round(T_REF / self.dt)))
        self.v = np.full(conn.n, V_REST, dtype=np.float64)
        self.i_syn = np.zeros(conn.n, dtype=np.float64)
        self.ref = np.zeros(conn.n, dtype=np.int32)
        # ring buffer of pending spikes: list of dest current additions per delay slot
        self.pending = [np.zeros(conn.n, dtype=np.float64) for _ in range(self.delay_steps)]
        self.ptr = 0
        self.silenced = np.zeros(conn.n, dtype=bool)
        self.stim_hz = np.zeros(conn.n, dtype=np.float64)
        self.spike_count = np.zeros(conn.n, dtype=np.int64)
        self.time = 0.0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.v[:] = V_REST
        self.i_syn[:] = 0.0
        self.ref[:] = 0
        for buf in self.pending:
            buf[:] = 0.0
        self.ptr = 0
        self.spike_count[:] = 0
        self.time = 0.0

    def stimulate(self, ids: list[int], hz: float = STIM_HZ) -> None:
        for i in ids:
            self.stim_hz[i] = hz

    def clear_stimulation(self) -> None:
        self.stim_hz[:] = 0.0

    def silence(self, ids: list[int]) -> None:
        for i in ids:
            self.silenced[i] = True

    def unsilence(self) -> None:
        self.silenced[:] = False

    def step(self) -> np.ndarray:
        dt = self.dt
        # Arrival of delayed spikes.
        self.i_syn += self.pending[self.ptr]
        self.pending[self.ptr][:] = 0.0
        # Synaptic decay and membrane leak.
        self.i_syn *= np.exp(-dt / TAU_SYN)
        # Poisson stimulation current as extra input spikes into those cells
        # (Shiu: stimulated neurons receive 150 Hz Poisson, strong enough to fire).
        if np.any(self.stim_hz):
            p = self.stim_hz * (dt / 1000.0)
            extra = self.rng.random(self.conn.n) < p
            # A stimulus event is treated as crossing threshold next.
        else:
            extra = np.zeros(self.conn.n, dtype=bool)
        dv = ((V_REST - self.v) / TAU_M + self.i_syn) * dt
        active = self.ref <= 0
        self.v[active] += dv[active]
        self.ref[self.ref > 0] -= 1
        fired = ((self.v >= V_THRESH) | extra) & (~self.silenced) & active
        self.v[fired] = V_REST
        self.ref[fired] = self.ref_steps
        self.spike_count += fired
        # Enqueue synaptic events
        if np.any(fired) and len(self.conn.src):
            src_fire = fired[self.conn.src]
            if np.any(src_fire):
                # add weight to dest at delay
                # next slot is (ptr + delay_steps) % delay_steps == ptr after we increment,
                # so we write into ptr (the buffer we just cleared will come back after a full cycle).
                # After increment, the write lands `delay_steps` steps later.
                dest = self.conn.dst[src_fire]
                w = self.conn.weight[src_fire]
                slot = self.pending[self.ptr]
                np.add.at(slot, dest, w)
        self.ptr = (self.ptr + 1) % self.delay_steps
        self.time += dt
        return fired

    def run(self, duration_ms: float) -> dict:
        n_steps = int(round(duration_ms / self.dt))
        t0 = self.time
        c0 = self.spike_count.copy()
        for _ in range(n_steps):
            self.step()
        elapsed = max(self.time - t0, 1e-9)
        counts = self.spike_count - c0
        rates = counts / (elapsed / 1000.0)  # Hz
        return {"duration_ms": elapsed, "counts": counts, "rates_hz": rates}

    def rate(self, ids: list[int], duration_ms: float) -> float:
        out = self.run(duration_ms)
        if not ids:
            return 0.0
        return float(np.mean(out["rates_hz"][np.asarray(ids)]))


def run_stimulus_protocol(
    conn: Connectome,
    stim_group: str,
    readout: str,
    duration_ms: float = 200.0,
    silence_group: str | None = None,
    shuffle: bool = False,
    seed: int = 0,
    dt: float = 0.1,
    stim_hz: float = STIM_HZ,
) -> dict:
    used = shuffle_wiring(conn, seed=seed) if shuffle else conn
    net = LIFNetwork(used, dt=dt, seed=seed)
    net.reset(seed=seed)
    if silence_group:
        net.silence(used.ids(silence_group))
    net.stimulate(used.ids(stim_group), hz=stim_hz)
    rate = net.rate(used.ids(readout), duration_ms=duration_ms)
    return {
        "stim_group": stim_group,
        "readout": readout,
        "rate_hz": rate,
        "shuffled": shuffle,
        "silenced": silence_group,
        "duration_ms": duration_ms,
    }
