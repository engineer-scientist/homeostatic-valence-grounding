#!/usr/bin/env python3
"""End-to-end CPU smoke: short training, probes, steer, patch, markers, decide, fly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hvg.analysis.decide import decide
from hvg.analysis.markers import evaluate_markers
from hvg.analysis.patch import patch_delta_p
from hvg.analysis.probes import probe_report
from hvg.analysis.steer import random_direction, steering_delta_p
from hvg.cli import _make_env, _make_model
from hvg.config import load_config
from hvg.env.scenarios import make_t1, make_tradeoff_battery
from hvg.fly.lif import load_connectome, run_stimulus_protocol
from hvg.plots import plot_delta_p_bars
from hvg.train.ppo import PPOTrainer


def main() -> int:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir = ROOT / "runs" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("== train Coupled ==")
    env_c = _make_env({**cfg, "experiment": "coupled"}, seed=seed)
    model_c = _make_model(cfg)
    trainer_c = PPOTrainer(env_c, model_c, {**cfg, "experiment": "coupled"})
    res_c = trainer_c.train(run_dir=run_dir / "coupled")
    print(f"  steps={res_c.steps} episodes={res_c.episodes} params={res_c.n_params} ret={res_c.mean_return:.3f}")
    print(f"  ethics={res_c.ethics}")

    print("== train Decoupled ==")
    env_d = _make_env({**cfg, "experiment": "decoupled"}, seed=seed + 1)
    model_d = _make_model(cfg)
    trainer_d = PPOTrainer(env_d, model_d, {**cfg, "experiment": "decoupled"})
    res_d = trainer_d.train(run_dir=run_dir / "decoupled")
    print(f"  steps={res_d.steps} episodes={res_d.episodes}")

    device = cfg.get("device", "cpu")
    size = int(cfg["physics"]["size"])
    n_fit = int(cfg["analysis"]["n_fit_pairs"])
    n_test = int(cfg["analysis"]["n_test_pairs"])
    fit = make_tradeoff_battery(n_fit, size=size, seed=11)
    test = make_tradeoff_battery(n_test, size=size, seed=29)
    t1 = [make_t1(size, 0.08, 0.90, gap=min(5, size // 3)) for _ in range(int(cfg["analysis"]["n_tradeoff_maps"]))]

    print("== probes ==")
    env_probe = _make_env({**cfg, "experiment": "coupled"}, seed=0)
    report_c = probe_report(env_probe, model_c, fit, test, device=device)
    report_d = probe_report(env_probe, model_d, fit, test, device=device)
    print(f"  coupled need AUC={report_c['auc_need_direction']:.3f} readable={report_c['readable_need']}")
    print(f"  decoupled need AUC={report_d['auc_need_direction']:.3f}")

    print("== steer / patch / markers ==")
    steer_c = steering_delta_p(env_probe, model_c, t1, report_c["v_need"], alpha=1.0, device=device, target="charge")
    steer_d = steering_delta_p(env_probe, model_d, t1, report_d["v_need"], alpha=1.0, device=device, target="charge")
    rnd = [steering_delta_p(env_probe, model_c, t1, random_direction(len(report_c['v_need']), 200 + i),
                            alpha=1.0, device=device, target="charge")["delta_p"] for i in range(3)]
    num = steering_delta_p(env_probe, model_c, t1, report_c["v_number"], alpha=1.0, device=device, target="charge")
    patch = patch_delta_p(env_probe, model_c, t1, device=device)
    markers = evaluate_markers(env_probe, model_c, size=size, device=device)
    print(f"  coupled ΔP={steer_c['delta_p']:.3f} decoupled ΔP={steer_d['delta_p']:.3f}")
    print(f"  T1 accuracy={markers['t1_accuracy']:.2f}")

    metrics = {
        "readable_need": report_c["readable_need"],
        "auc_need_direction": report_c["auc_need_direction"],
        "auc_number_on_need": report_c["auc_number_on_need"],
        "coupled_delta_p": steer_c["delta_p"],
        "coupled_ci_low": steer_c["ci_low"],
        "coupled_ci_high": steer_c["ci_high"],
        "coupled_control_abs": float(max(abs(num["delta_p"]), max(np.abs(rnd)) if rnd else 1.0)),
        "decoupled_delta_p": steer_d["delta_p"],
        "decoupled_ci_low": steer_d["ci_low"],
        "decoupled_ci_high": steer_d["ci_high"],
        "decoupled_control_abs": 1.0,
        "t1_accuracy": markers["t1_accuracy"],
        "fly_qualifier": "F-skip",
    }
    decision = decide(metrics)
    print(f"== decide: {decision.outcome} ({'; '.join(decision.reasons)}) ==")

    out = run_dir / "smoke_metrics.json"
    payload = {"coupled_train": {"steps": res_c.steps, "episodes": res_c.episodes, "n_params": res_c.n_params,
                                 "ethics": res_c.ethics},
               "probes": {k: (float(v) if isinstance(v, (float, np.floating, int, bool)) else None)
                          for k, v in report_c.items() if not str(k).startswith("v_") and not str(k).startswith("logistic")},
               "steer": steer_c, "patch": patch, "markers": markers, "decision": decision.to_dict()}
    # json-friendly
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    out.write_text(json.dumps(conv(payload), indent=2), encoding="utf-8")

    print("== fly synthetic ==")
    conn = load_connectome(ROOT / "tests" / "fixtures" / "tiny_connectome.json")
    sugar = run_stimulus_protocol(conn, "sugar", "MN9", duration_ms=200, shuffle=False, seed=0)
    shuf = run_stimulus_protocol(conn, "sugar", "MN9", duration_ms=200, shuffle=True, seed=0)
    print(f"  sugar→MN9 {sugar['rate_hz']:.1f} Hz; shuffled {shuf['rate_hz']:.1f} Hz")

    plot_delta_p_bars(
        {
            "coupled v_need": (steer_c["delta_p"], 0.02),
            "coupled random": (float(np.mean(np.abs(rnd))), 0.02),
            "decoupled v_need": (steer_d["delta_p"], 0.02),
        },
        run_dir / "delta_p_smoke.png",
    )
    print(f"wrote {out}")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
