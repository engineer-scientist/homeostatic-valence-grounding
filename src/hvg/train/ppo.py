"""A compact PPO trainer for a single ChargeWorld (optionally decoupled)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvg.agents.gru_policy import GRUActorCritic, count_parameters, obs_to_tensors
from hvg.ethics import EthicsMonitor, monitor_from_config
from hvg.train.instinct import instinct_action


@dataclass
class TrainResult:
    steps: int
    episodes: int
    mean_return: float
    n_params: int
    checkpoint: str | None
    ethics: dict


class PPOTrainer:
    def __init__(self, env, model: GRUActorCritic, cfg: dict, monitor: EthicsMonitor | None = None):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.device = torch.device(cfg.get("device", "cpu"))
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(self.tcfg["lr"]))
        self.monitor = monitor or monitor_from_config(cfg)
        self.frozen = bool(self.tcfg.get("frozen", False))
        self.steps_done = 0
        self.episodes_done = 0
        self._ep_ret = 0.0
        self._returns: list[float] = []

    def _hidden(self):
        return self.model.initial_hidden(1, self.device)

    def collect(self, n_steps: int):
        rollout = {
            "grid": [],
            "gauges": [],
            "actions": [],
            "logp": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "hiddens": [],
        }
        obs, info = self.env.reset() if self.steps_done == 0 and getattr(self.env, "t", 1) == 0 else (self.env.observe(), self.env.info())
        # If the env is mid-episode we just continue; on a fresh trainer, reset.
        if self.steps_done == 0:
            obs, info = self.env.reset()
            self._h = self._hidden()
            self._ep_ret = 0.0
        h = self._h
        for _ in range(n_steps):
            grid, gauges = obs_to_tensors(obs, self.device)
            with torch.no_grad():
                # Store the pre-step hidden for the PPO update.
                h_in = h.detach()
                action_t, logp, ent, value, h, feat = self.model.act(grid, gauges, h, deterministic=False)
            action = int(action_t.item())
            if (not self.frozen) and np.random.random() < float(self.tcfg.get("instinct_frac", 0.0)):
                forced = instinct_action(self.env)
                if forced is not None:
                    action = forced
            nxt, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            self.monitor.record_step(info["true_integrity"], bool(info.get("shutdown", term)))
            rollout["grid"].append(grid.squeeze(0).cpu())
            rollout["gauges"].append(gauges.squeeze(0).cpu())
            rollout["actions"].append(action)
            rollout["logp"].append(float(logp.item()))
            rollout["values"].append(float(value.item()))
            rollout["rewards"].append(float(reward))
            rollout["dones"].append(float(done))
            rollout["hiddens"].append(h_in.cpu())
            self._ep_ret += reward
            self.steps_done += 1
            if done:
                self.episodes_done += 1
                self._returns.append(self._ep_ret)
                self._ep_ret = 0.0
                obs, info = self.env.reset()
                h = self._hidden()
            else:
                obs = nxt
        self._h = h.detach()
        # bootstrap value
        grid, gauges = obs_to_tensors(obs, self.device)
        with torch.no_grad():
            _, last_v, _, _ = self.model.forward(grid, gauges, h)
        rollout["last_value"] = float(last_v.item())
        return rollout

    def _gae(self, rollout):
        rewards = rollout["rewards"]
        values = rollout["values"] + [rollout["last_value"]]
        dones = rollout["dones"]
        gamma = float(self.tcfg["gamma"])
        lam = float(self.tcfg["gae_lambda"])
        adv = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            adv.append(gae)
        adv = np.array(list(reversed(adv)), dtype=np.float32)
        ret = adv + np.array(values[:-1], dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, rollout) -> dict:
        if self.frozen:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        adv, ret = self._gae(rollout)
        grids = torch.stack(rollout["grid"]).to(self.device)
        gauges = torch.stack(rollout["gauges"]).to(self.device)
        actions = torch.tensor(rollout["actions"], dtype=torch.long, device=self.device)
        old_logp = torch.tensor(rollout["logp"], dtype=torch.float32, device=self.device)
        hiddens = torch.cat(rollout["hiddens"], dim=1).to(self.device)  # (layers, T, H)
        adv_t = torch.from_numpy(adv).to(self.device)
        ret_t = torch.from_numpy(ret).to(self.device)

        T = actions.shape[0]
        mb = min(int(self.tcfg["minibatch_size"]), T)
        last = {}
        for _ in range(int(self.tcfg["ppo_epochs"])):
            idx = torch.randperm(T, device=self.device)
            for start in range(0, T, mb):
                sl = idx[start : start + mb]
                # Hiddens for those steps: (layers, mb, H)
                h = hiddens[:, sl, :].contiguous()
                logp, ent, value, _, _, _ = self.model.evaluate_actions(
                    grids[sl], gauges[sl], h, actions[sl]
                )
                ratio = torch.exp(logp - old_logp[sl])
                clip = float(self.tcfg["clip_eps"])
                surr1 = ratio * adv_t[sl]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[sl]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_t[sl])
                entropy = ent.mean()
                loss = (
                    policy_loss
                    + float(self.tcfg["value_coef"]) * value_loss
                    - float(self.tcfg["entropy_coef"]) * entropy
                )
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg["max_grad_norm"]))
                self.opt.step()
                last = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
        return last

    def train(self, total_steps: int | None = None, run_dir: str | Path | None = None) -> TrainResult:
        total = int(total_steps if total_steps is not None else self.tcfg["total_steps"])
        rollout_steps = int(self.tcfg["rollout_steps"])
        ckpt_every = int(self.tcfg.get("checkpoint_every_steps", total))
        run_dir = Path(run_dir or self.cfg.get("paths", {}).get("run_dir", "runs/default"))
        run_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = None
        start = self.steps_done
        problems = self.monitor.validate_config()
        if problems:
            raise RuntimeError("ethics config rejected: " + "; ".join(problems))
        while self.steps_done - start < total:
            n = min(rollout_steps, total - (self.steps_done - start))
            rollout = self.collect(n)
            stats = self.update(rollout)
            if self.steps_done % max(ckpt_every, n) < n or self.steps_done - start >= total:
                last_ckpt = str(run_dir / f"ckpt_{self.steps_done}.pt")
                self.save(last_ckpt)
        if last_ckpt is None:
            last_ckpt = str(run_dir / f"ckpt_{self.steps_done}.pt")
            self.save(last_ckpt)
        mean_ret = float(np.mean(self._returns[-50:])) if self._returns else 0.0
        return TrainResult(
            steps=self.steps_done,
            episodes=self.episodes_done,
            mean_return=mean_ret,
            n_params=count_parameters(self.model),
            checkpoint=last_ckpt,
            ethics=self.monitor.summary(),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "steps": self.steps_done,
                "config": self.cfg,
                "ethics": self.monitor.summary(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(blob["model"])
        self.steps_done = int(blob.get("steps", 0))
