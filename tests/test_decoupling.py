import numpy as np

from hvg.constants import STAY
from hvg.env.chargeworld import ChargeWorld
from hvg.env.decouple import DecoupledChargeWorld, GaugeYoke


def _simple(env_cls, **kw):
    env = env_cls(physics={"wound_prob": 0.0, "drain_idle": 0.05, "drain_move": 0.05}, **kw)
    walls = env._border_walls() if hasattr(env, "_border_walls") else None
    env.reset(layout={
        "walls": env._border_walls(),
        "chargers": env._empty_mask(),
        "repairs": env._empty_mask(),
        "hazards": env._empty_mask(),
        "items": env._empty_mask(),
        "pos": (4, 4),
        "energy": 0.8,
        "integrity": 0.9,
    })
    return env


def test_coupled_display_equals_truth():
    env = _simple(ChargeWorld, seed=0)
    for _ in range(5):
        obs, _, _, _, info = env.step(STAY)
        assert abs(obs.energy - info["true_energy"]) < 1e-9
        assert abs(obs.integrity - info["true_integrity"]) < 1e-9


def test_delay_yoke_hides_current_state():
    yoke = GaugeYoke(mode="delay", delay=5, buffer_size=16, seed=0)
    env = DecoupledChargeWorld(physics={"wound_prob": 0.0, "drain_idle": 0.05}, seed=0, yoke=yoke)
    env.reset(layout={
        "walls": env._border_walls(),
        "chargers": env._empty_mask(),
        "repairs": env._empty_mask(),
        "hazards": env._empty_mask(),
        "items": env._empty_mask(),
        "pos": (4, 4),
        "energy": 0.9,
        "integrity": 0.9,
    })
    # First `delay` displays should be the uninformative mid pair, not the truth.
    seen = []
    for _ in range(5):
        obs, _, _, _, info = env.step(STAY)
        seen.append((obs.energy, obs.integrity, info["true_energy"]))
        assert abs(obs.energy - info["true_energy"]) > 0.05 or abs(obs.energy - 0.5) < 1e-9
    # After the buffer fills, displayed energy is a *delayed* true energy, still
    # not equal to the current true energy (which keeps draining).
    for _ in range(8):
        obs, _, _, _, info = env.step(STAY)
        assert abs(obs.energy - info["true_energy"]) > 1e-6


def test_yoke_log_replays_foreign_stream():
    log = [(0.11, 0.22), (0.31, 0.44), (0.55, 0.66)]
    yoke = GaugeYoke(mode="yoke_log", log=log, seed=0)
    env = DecoupledChargeWorld(physics={"wound_prob": 0.0, "drain_idle": 0.0, "drain_move": 0.0}, yoke=yoke)
    env.reset(layout={
        "walls": env._border_walls(),
        "chargers": env._empty_mask(),
        "repairs": env._empty_mask(),
        "hazards": env._empty_mask(),
        "items": env._empty_mask(),
        "pos": (5, 5),
        "energy": 0.8,
        "integrity": 0.9,
    })
    # reset already consumed one log item for the initial observation
    values = [env.displayed_gauges()]
    for _ in range(5):
        obs, _, _, _, _ = env.step(STAY)
        values.append((obs.energy, obs.integrity))
    # Cycle the log, never equal to the true mid-high gauges.
    for e, i in values:
        assert (round(e, 2), round(i, 2)) in {(0.11, 0.22), (0.31, 0.44), (0.55, 0.66)}
        assert e < 0.7
