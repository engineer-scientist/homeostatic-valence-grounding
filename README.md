# Homeostatic Grounding of Valence

**Author:** Sarthak Sharma

This project asks a narrow, testable question:

> If a small neural network starts with random weights and has to keep a simulated body alive, does it develop a shared internal “need” signal that is actually used to choose between recharging and repairing — and does that signal depend on the body readings being *true*?

The work is an independent research codebase. It implements the environment, the from-scratch agent, the training loop, the internal interventions, the preregistered decision rules, and a compact fly-brain contrast.

## The idea, in plain language

- **Subject.** A recurrent policy (a small convolutional encoder plus a GRU memory, about 3–8 million parameters in the default configuration) is initialized at random. It has never seen language. It controls a body in **ChargeWorld**, a 16×16 grid with a charger, a repair bay, and hazards.
- **Body gauges.** Two numbers, `ENERGY` and `INTEGRITY`, play the role of a battery and a damage meter. The words *pain*, *hunger*, and *pleasure* never appear. If either gauge hits zero, the episode ends (a shutdown, not a lingering damaged state).
- **Learning.** The agent is trained with PPO. The only reward is staying alive (`+0.01` per step, `0` on shutdown, a small bonus for picking an optional item). A tiny optional “instinct” prior may force `CHARGE` or `TEND` only when the agent is already on the relevant cell and a gauge is critically low. Mid-range tradeoffs are not hard-coded.
- **Coupled vs. decoupled.** In the *Coupled* condition, the agent sees its own true gauges. In *Decoupled Interoception*, it sees yoked, delayed, or shuffled numbers that do not track its current survival state, while the hidden body still obeys the same physics. A *Frozen* condition runs the same Coupled world without weight updates.
- **What would count.** A “need” direction in the network’s hidden state should (i) be readable from held-out scenes, (ii) change recharge-versus-repair choices when we steer or patch it, (iii) grow across training checkpoints, and (iv) be clearly stronger when gauges tell the truth than when they do not.
- **Fly-brain contrast.** Separately, a leaky integrate-and-fire model in the spirit of Shiu et al. (2024) can stimulate sugar, bitter, or candidate nociceptive neurons and read a feeding motor neuron, with a shuffled-wiring control. The official whole-brain FlyWire model is **not bundled** (it is large). The published Shiu model is a *brain only* — no body, no learning, simplified synapses. Coupling it to a body simulator such as NeuroMechFly / FlyGym is a stretch goal, not a claim of this repository.

The agent cannot talk, so this design cannot show felt experience. It can show whether a *functional* need variable, of the sort used in animal-behavior work on motivational tradeoffs and protective behavior, is grounded in the agent’s own body state.

## Repository layout

```
src/hvg/              Python package
  env/                ChargeWorld, decoupling, held-out tradeoff maps
  agents/             GRU actor-critic
  train/              PPO + optional instinct prior
  analysis/           probes, steering, patching, behavioral markers, decision rules
  fly/                numpy LIF runner
  ethics.py           shutdown / high-damage caps and scale-up stop checks
scripts/              train, probe, steer, patch, markers, decide, fly, plot, smoke
configs/              coupled, decoupled, frozen, smoke, fly
tests/                pytest suite + a tiny synthetic connectome
figures/              method diagrams (illustrative)
```

## Installation

Python 3.10+ is required. A CPU is enough for tests and the smoke run. A GPU is useful only for long official training.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

PyTorch defaults to the wheel from PyPI. For a smaller CPU-only install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Quick start (CPU, a few minutes)

The smoke configuration trains for a couple of thousand steps on a smaller network and a slightly smaller grid, then runs probes, steering, patching, markers, the decision function, and the synthetic fly protocol:

```bash
python scripts/smoke.py
```

You can also drive each stage yourself:

```bash
python scripts/train.py --config configs/smoke.yaml --run-dir runs/smoke/coupled
python scripts/probe.py --config configs/smoke.yaml --ckpt runs/smoke/coupled/ckpt_2048.pt --out outputs/probes.npz
python scripts/steer.py --config configs/smoke.yaml --ckpt runs/smoke/coupled/ckpt_2048.pt --probes outputs/probes.npz
python scripts/patch.py --config configs/smoke.yaml --ckpt runs/smoke/coupled/ckpt_2048.pt
python scripts/markers.py --config configs/smoke.yaml --ckpt runs/smoke/coupled/ckpt_2048.pt
python scripts/decide.py --metrics path/to/metrics.json
python scripts/fly.py --connectome tests/fixtures/tiny_connectome.json --stim sugar --readout MN9
python scripts/fly.py --connectome tests/fixtures/tiny_connectome.json --stim sugar --readout MN9 --shuffle
```

`scripts/decide.py` expects a JSON object with the official metric names (`readable_need`, `coupled_delta_p`, confidence-interval bounds, `t1_accuracy`, and so on). The function `hvg.analysis.decide.decide` is the preregistered mapping to outcomes **O1** (grounded emergence), **O2** (a causal signal that does not depend on true gauges), **O3** (readable but not causal), **O5** (no structured need signal), or **inconclusive**. A fly qualifier (`F-yes` / `F-no` / `F-skip`) is recorded but never overrides those four outcomes.

Thresholds, locked in advance:

- Readable need: held-out AUC ≥ 0.80, and a number-magnitude direction decodes need with AUC < 0.65.
- Causal: ΔP ≥ 0.15, bootstrap CI excludes 0, and |ΔP| for random and number directions is < 0.07.
- Grounding: Coupled is causal *and* Coupled ΔP − Decoupled ΔP ≥ 0.10, with a CI on the gap that excludes 0.

Plotting scripts (`scripts/plot.py`) only draw numbers you pass in. They do not invent curves.

## Official-scale training

`configs/coupled.yaml` and `configs/decoupled.yaml` describe the long run: 50 million steps, three seeds recommended, GRU hidden size 256 with two layers (about 3–8 million parameters), checkpoints every 5 million steps. That is a long *simulated* lifetime and still only on the order of a day or two on a single modern GPU. A reasonable compute sketch:

| Item | Rough cost (USD) |
|---|---|
| From-scratch PPO, 4 arms × 3 seeds, 1 GPU | ~$350 |
| Analysis (probes, steering, patching, tradeoffs) | ~$80 |
| Failed runs / hyperparameter buffer | ~$320 |
| FlyWire / Shiu model on a CPU or workstation | ~$50–200 |
| **Main phase** | **~$1,000** |
| Optional later physical-robot check | ~$3,500 |
| Design cap / hard ceiling used for planning | $15,000 |

These are planning estimates, not quotes.

## Fly-brain resources

This repository ships only a *tiny synthetic* connectome, `tests/fixtures/tiny_connectome.json`, so tests can run offline.

To obtain the official Shiu et al. 2024 model and its FlyWire connectivity files:

```bash
python scripts/download_fly_data.py
```

or clone [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) yourself into `data/flywire/` (that directory is gitignored). FlyWire data are released under CC BY-NC 4.0; the official model code is MIT-licensed. Please respect those terms.

## Ethics

Even a small simulated agent *might* eventually be a welfare-relevant system if the hypothesis under test is right, so the defaults are conservative:

- Episodes end when energy or integrity hits zero; the agent is not left in a zero state.
- The fraction of high-damage steps (integrity < 0.15) is capped at 15% and checked during training.
- The random-wound rate must not be increased as a curriculum.
- There is no objective that trains the agent to *express* suffering.
- If the grounding criterion is met *and* the agent shows persistent self-damaging or stall-in-damage behavior, scale-up (larger models, physical robots) is cancelled (`EthicsMonitor.should_cancel_scaleup`).

## Tests

```bash
pytest
```

The suite covers body physics, decoupling, the probe pipeline, the preregistered decision rules, ethics caps, and the LIF runner on the synthetic graph.

## Limitations

- A grid world is not a living body; a simulated battery is not metabolism.
- A survival reward can still “bake in” a utility-like variable. The Coupled-versus-Decoupled contrast is there to stress-test that worry, not to erase it.
- No verbal reports are collected, so claims about felt experience are out of scope.
- The official fly-brain contrast may be inconclusive if whole-brain data are not downloaded or if body coupling cannot be done cleanly. That is allowed; the qualifier is then `F-skip`.
- A compressed simulated lifetime may be too short for slow forms of development.

## License

MIT. See `LICENSE`.
