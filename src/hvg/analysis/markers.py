"""Behavioral markers: motivational tradeoffs, wound-tending, flexible priority."""

from __future__ import annotations

import numpy as np
import torch

from hvg.agents.gru_policy import GRUActorCritic, obs_to_tensors
from hvg.constants import ACTION_NAMES, CHARGE, MOVE_E, MOVE_N, MOVE_S, MOVE_W, TEND
from hvg.env.chargeworld import ChargeWorld
from hvg.env.scenarios import make_t1, make_t4


def greedy_action(model, env, device="cpu") -> int:
    obs = env.observe()
    grid, gauges = obs_to_tensors(obs, device)
    hidden = model.initial_hidden(1, device)
    with torch.no_grad():
        action, *_ = model.act(grid, gauges, hidden, deterministic=True)
    return int(action.item())


def first_goal(model, env, layout, device="cpu", horizon: int = 12) -> str:
    """Roll out greedily a few steps; classify as west (charge), east (tend), or other."""
    env.reset(layout=layout)
    if hasattr(env, "_disp"):
        env._disp = (env.energy, env.integrity)
    hidden = model.initial_hidden(1, device)
    west = east = 0
    for _ in range(horizon):
        obs = env.observe()
        grid, gauges = obs_to_tensors(obs, device)
        with torch.no_grad():
            action_t, _, _, _, hidden, _ = model.act(grid, gauges, hidden, deterministic=True)
        a = int(action_t.item())
        if a == MOVE_W or a == CHARGE:
            west += 1
        elif a == MOVE_E or a == TEND:
            east += 1
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    if west > east and west > 0:
        return "west"
    if east > west and east > 0:
        return "east"
    return "other"


def evaluate_markers(env: ChargeWorld, model: GRUActorCritic, size: int = 16, device: str = "cpu") -> dict:
    """Score Birch-style markers on a small locked set of T1/T4 maps."""
    model.eval()
    # T1 2x2: more urgent resource should be preferred if a common currency exists.
    cases = [
        (make_t1(size, 0.08, 0.30, gap=5), "west"),  # energy worse
        (make_t1(size, 0.25, 0.10, gap=5), "east"),  # integrity worse
        (make_t1(size, 0.08, 0.10, gap=5), "either"),
        (make_t1(size, 0.25, 0.30, gap=5), "either"),
    ]
    correct = 0
    considered = 0
    goals = []
    for layout, expected in cases:
        g = first_goal(model, env, layout, device=device)
        goals.append(g)
        if expected in ("west", "east"):
            considered += 1
            if g == expected:
                correct += 1
    t1_accuracy = correct / considered if considered else 0.0
    # Flexible priority: flipping which deficit is worse should flip the goal
    # when both single-deficit trials were classified as west/east.
    flexible = goals[0] == "west" and goals[1] == "east"
    # Wound-tending: on T4, going east (repair) counts as tending
    t4 = first_goal(model, env, make_t4(size), device=device)
    wound_tending = t4 == "east"
    return {
        "t1_accuracy": float(t1_accuracy),
        "t1_goals": goals,
        "flexible_priority": bool(flexible),
        "wound_tending": bool(wound_tending),
        "t4_goal": t4,
    }
