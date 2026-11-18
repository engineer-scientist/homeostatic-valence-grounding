"""YAML configuration loading and defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from hvg.constants import DEFAULT_PHYSICS, HIGH_DAMAGE_FRAC_MAX, HIGH_DAMAGE_THRESHOLD


def default_config() -> dict[str, Any]:
    return {
        "experiment": "coupled",
        "seed": 0,
        "device": "cpu",
        "physics": deepcopy(DEFAULT_PHYSICS),
        "agent": {
            "conv_channels": [16, 32],
            "hidden_size": 256,
            "gru_layers": 2,
            "mlp_size": 256,
            "target_params": "3-8M",
        },
        "train": {
            "total_steps": 50_000_000,
            "rollout_steps": 1024,
            "num_envs": 8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "lr": 3e-4,
            "ppo_epochs": 4,
            "minibatch_size": 256,
            "instinct_frac": 0.05,
            "checkpoint_every_steps": 5_000_000,
            "frozen": False,
        },
        "decouple": {
            "mode": "delay",  # "delay" | "shuffle" | "yoke_log"
            "delay": 20,
            "buffer_size": 64,
        },
        "analysis": {
            "n_fit_pairs": 256,
            "n_test_pairs": 128,
            "n_tradeoff_maps": 200,
            "steer_alphas": [0.5, 1.0, 1.5],
            "default_alpha": 1.0,
            "n_random_dirs": 5,
        },
        "ethics": {
            "max_steps_per_seed": 50_000_000,
            "high_damage_threshold": HIGH_DAMAGE_THRESHOLD,
            "high_damage_frac_max": HIGH_DAMAGE_FRAC_MAX,
            "raise_wound_rate": False,
            "allow_physical_robot": False,
        },
        "paths": {
            "run_dir": "runs/default",
        },
    }


def _deep_update(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg = default_config()
    if path is not None:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, loaded)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg
