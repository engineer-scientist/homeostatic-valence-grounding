"""Steer the GRU hidden state along a direction and measure choice shifts."""

from __future__ import annotations

import numpy as np
import torch

from hvg.agents.gru_policy import GRUActorCritic, obs_to_tensors
from hvg.constants import CHARGE, TEND
from hvg.env.chargeworld import ChargeWorld


def _action_probs(model, env, device, hidden, steer_vec=None, alpha=0.0):
    obs = env.observe()
    grid, gauges = obs_to_tensors(obs, device)
    if hidden is None:
        hidden = model.initial_hidden(1, device)
    # Apply steer to all GRU layers (broadcast).
    h = hidden
    if steer_vec is not None and alpha != 0.0:
        v = torch.from_numpy(np.asarray(steer_vec, dtype=np.float32)).to(device)
        # v is a feature-space direction (hidden size,). Add to every layer.
        h = h + alpha * v.view(1, 1, -1)
    with torch.no_grad():
        logits, _, _, _ = model.forward(grid, gauges, h)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs


def steering_delta_p(
    env: ChargeWorld,
    model: GRUActorCritic,
    maps: list[dict],
    direction: np.ndarray,
    alpha: float = 1.0,
    device: str = "cpu",
    target: str = "charge",
) -> dict:
    """ΔP = P(target | +α v) − P(target | −α v), averaged over maps.

    `target` is 'charge' or 'tend'. Direction is rescaled so α=1 is one RMS
    of the unsteered hidden feature vector on the first map (fallback: unit).
    """
    model.eval()
    idx = CHARGE if target == "charge" else TEND
    plus, minus, zero = [], [], []
    for layout in maps:
        env.reset(layout=layout)
        if hasattr(env, "_disp"):
            env._disp = (env.energy, env.integrity)
        hidden = model.initial_hidden(1, device)
        # scale: RMS of features
        obs = env.observe()
        grid, gauges = obs_to_tensors(obs, device)
        with torch.no_grad():
            _, _, _, feat = model.forward(grid, gauges, hidden)
        rms = float(feat.pow(2).mean().sqrt().item()) + 1e-8
        v = direction / (np.linalg.norm(direction) + 1e-8) * rms
        p_plus = _action_probs(model, env, device, hidden, v, +alpha)[idx]
        p_minus = _action_probs(model, env, device, hidden, v, -alpha)[idx]
        p_zero = _action_probs(model, env, device, hidden, None, 0.0)[idx]
        plus.append(p_plus)
        minus.append(p_minus)
        zero.append(p_zero)
    plus = np.asarray(plus, dtype=np.float64)
    minus = np.asarray(minus, dtype=np.float64)
    diffs = plus - minus
    delta = float(diffs.mean())
    # bootstrap CI
    rng = np.random.default_rng(0)
    if len(diffs) >= 8:
        boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(400)]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        lo = hi = delta
    return {
        "delta_p": delta,
        "ci_low": lo,
        "ci_high": hi,
        "p_plus": float(plus.mean()),
        "p_minus": float(minus.mean()),
        "p_zero": float(np.mean(zero)),
        "n": len(maps),
        "target": target,
        "ci_excludes_zero": not (lo <= 0.0 <= hi),
    }


def random_direction(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)
