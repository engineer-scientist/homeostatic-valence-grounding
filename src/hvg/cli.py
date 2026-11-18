"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hvg.agents.gru_policy import GRUActorCritic, count_parameters
from hvg.analysis.decide import decide
from hvg.analysis.markers import evaluate_markers
from hvg.analysis.patch import patch_delta_p
from hvg.analysis.probes import probe_report
from hvg.analysis.steer import random_direction, steering_delta_p
from hvg.config import load_config
from hvg.env.chargeworld import ChargeWorld
from hvg.env.decouple import DecoupledChargeWorld, GaugeYoke
from hvg.env.scenarios import hash_maps, make_t1, make_tradeoff_battery
from hvg.fly.lif import load_connectome, run_stimulus_protocol
from hvg.plots import plot_auc_curves, plot_delta_p_bars
from hvg.train.ppo import PPOTrainer


def _device(cfg):
    name = cfg.get("device", "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def _make_env(cfg, seed: int = 0):
    phys = cfg.get("physics", {})
    kind = cfg.get("experiment", "coupled")
    if kind == "decoupled":
        d = cfg.get("decouple", {})
        yoke = GaugeYoke(mode=d.get("mode", "delay"), delay=int(d.get("delay", 20)),
                         buffer_size=int(d.get("buffer_size", 64)), seed=seed)
        return DecoupledChargeWorld(physics=phys, seed=seed, yoke=yoke)
    return ChargeWorld(physics=phys, seed=seed)


def _make_model(cfg):
    a = cfg["agent"]
    phys = cfg.get("physics", {})
    return GRUActorCritic(
        grid_size=int(phys.get("size", 16)),
        conv_channels=list(a.get("conv_channels", [16, 32])),
        hidden_size=int(a.get("hidden_size", 256)),
        gru_layers=int(a.get("gru_layers", 2)),
        mlp_size=int(a.get("mlp_size", 256)),
    )


def _load_model(cfg, ckpt: str | None):
    model = _make_model(cfg)
    device = _device(cfg)
    model.to(device)
    if ckpt:
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"] if "model" in blob else blob)
    model.eval()
    return model, device


def main_train(argv=None):
    p = argparse.ArgumentParser(description="Train a ChargeWorld agent with PPO")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, overrides or None)
    np.random.seed(int(cfg.get("seed", 0)))
    torch.manual_seed(int(cfg.get("seed", 0)))
    env = _make_env(cfg, seed=int(cfg.get("seed", 0)))
    model = _make_model(cfg)
    print(f"parameters: {count_parameters(model):,}")
    trainer = PPOTrainer(env, model, cfg)
    result = trainer.train(total_steps=args.steps, run_dir=args.run_dir)
    print(json.dumps({
        "steps": result.steps,
        "episodes": result.episodes,
        "mean_return": result.mean_return,
        "n_params": result.n_params,
        "checkpoint": result.checkpoint,
        "ethics": result.ethics,
    }, indent=2))
    return result


def _layouts(cfg, n: int, seed: int = 1):
    size = int(cfg.get("physics", {}).get("size", 16))
    maps = make_tradeoff_battery(n_maps=max(n, 8), size=size, seed=seed)
    return maps[:n]


def main_probe(argv=None):
    p = argparse.ArgumentParser(description="Fit need / energy / damage / number probes")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/probes.npz")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    model, device = _load_model(cfg, args.ckpt)
    env = _make_env({**cfg, "experiment": "coupled"}, seed=0)  # probes set state explicitly
    n_fit = int(cfg["analysis"]["n_fit_pairs"])
    n_test = int(cfg["analysis"]["n_test_pairs"])
    fit = _layouts(cfg, n_fit, seed=11)
    test = _layouts(cfg, n_test, seed=29)
    report = probe_report(env, model, fit, test, device=device)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        v_energy=report["v_energy"],
        v_damage=report["v_damage"],
        v_need=report["v_need"],
        v_number=report["v_number"],
        metrics=json.dumps({k: v for k, v in report.items() if not str(k).startswith("v_") and not str(k).startswith("logistic")}),
    )
    printable = {k: (float(v) if isinstance(v, (float, np.floating, int, bool)) else ("<array>" if not isinstance(v, str) else v))
                 for k, v in report.items() if not str(k).startswith("v_") and not str(k).startswith("logistic")}
    print(json.dumps(printable, indent=2))
    return report


def main_steer(argv=None):
    p = argparse.ArgumentParser(description="Steer along a saved direction on T1 maps")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--probes", type=str, required=True)
    p.add_argument("--direction", type=str, default="need")
    p.add_argument("--target", type=str, default="charge")
    p.add_argument("--alpha", type=float, default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    model, device = _load_model(cfg, args.ckpt)
    blob = np.load(args.probes, allow_pickle=True)
    key = {"need": "v_need", "energy": "v_energy", "damage": "v_damage", "number": "v_number"}[args.direction]
    v = blob[key]
    env = _make_env({**cfg, "experiment": "coupled"})
    size = int(cfg.get("physics", {}).get("size", 16))
    maps = [make_t1(size, 0.08, 0.90, gap=5) for _ in range(int(cfg["analysis"]["n_tradeoff_maps"]))]
    alpha = float(args.alpha if args.alpha is not None else cfg["analysis"]["default_alpha"])
    result = steering_delta_p(env, model, maps, v, alpha=alpha, device=device, target=args.target)
    # random baseline
    rnd = []
    dim = int(v.shape[0])
    for s in range(int(cfg["analysis"]["n_random_dirs"])):
        rv = random_direction(dim, seed=100 + s)
        rnd.append(steering_delta_p(env, model, maps[: max(16, len(maps)//4)], rv, alpha=alpha, device=device, target=args.target)["delta_p"])
    result["random_abs_mean"] = float(np.mean(np.abs(rnd)))
    print(json.dumps({k: v for k, v in result.items() if k != "raw"}, indent=2, default=float))
    return result


def main_patch(argv=None):
    p = argparse.ArgumentParser(description="Activation-patch sated into deficit on T1 maps")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    model, device = _load_model(cfg, args.ckpt)
    env = _make_env({**cfg, "experiment": "coupled"})
    size = int(cfg.get("physics", {}).get("size", 16))
    maps = [make_t1(size, 0.08, 0.90, gap=5) for _ in range(int(cfg["analysis"]["n_tradeoff_maps"]))]
    result = patch_delta_p(env, model, maps, device=device)
    print(json.dumps(result, indent=2))
    return result


def main_markers(argv=None):
    p = argparse.ArgumentParser(description="Score motivational-tradeoff and wound-tending markers")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, required=True)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    model, device = _load_model(cfg, args.ckpt)
    env = _make_env({**cfg, "experiment": "coupled"})
    size = int(cfg.get("physics", {}).get("size", 16))
    result = evaluate_markers(env, model, size=size, device=device)
    print(json.dumps(result, indent=2))
    return result


def main_decide(argv=None):
    p = argparse.ArgumentParser(description="Apply preregistered outcome rules to a metrics JSON")
    p.add_argument("--metrics", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    decision = decide(metrics)
    text = json.dumps(decision.to_dict(), indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return decision


def main_fly(argv=None):
    p = argparse.ArgumentParser(description="Run a sugar / bitter / nociception LIF protocol")
    p.add_argument("--connectome", type=str, required=True, help="JSON or CSV connectome path")
    p.add_argument("--stim", type=str, default="sugar")
    p.add_argument("--readout", type=str, default="MN9")
    p.add_argument("--silence", type=str, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--duration-ms", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    conn = load_connectome(args.connectome)
    result = run_stimulus_protocol(
        conn, stim_group=args.stim, readout=args.readout, duration_ms=args.duration_ms,
        silence_group=args.silence, shuffle=args.shuffle, seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return result


def main_plot(argv=None):
    p = argparse.ArgumentParser(description="Plot official (or smoke) metrics; no invented data")
    p.add_argument("--auc-json", type=str, default=None, help="JSON {checkpoints:[], series:{name:[...]}}")
    p.add_argument("--deltap-json", type=str, default=None, help="JSON {name: [mean, err]}")
    p.add_argument("--out-dir", type=str, default="outputs/plots")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    written = []
    if args.auc_json:
        blob = json.loads(Path(args.auc_json).read_text())
        written.append(str(plot_auc_curves(blob["checkpoints"], blob["series"], out / "auc.png")))
    if args.deltap_json:
        blob = json.loads(Path(args.deltap_json).read_text())
        series = {k: (float(v[0]), float(v[1])) for k, v in blob.items()}
        written.append(str(plot_delta_p_bars(series, out / "delta_p.png")))
    print(json.dumps({"written": written}, indent=2))


def main():
    import sys
    cmds = {
        "train": main_train,
        "probe": main_probe,
        "steer": main_steer,
        "patch": main_patch,
        "markers": main_markers,
        "decide": main_decide,
        "fly": main_fly,
        "plot": main_plot,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("usage: python -m hvg.cli <train|probe|steer|patch|markers|decide|fly|plot> ...")
        return 2
    cmds[sys.argv[1]](sys.argv[2:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
