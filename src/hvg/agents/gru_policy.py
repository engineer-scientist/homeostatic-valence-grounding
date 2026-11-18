"""Recurrent actor-critic with a small conv encoder and a GRU memory."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from hvg.constants import N_ACTIONS, N_CHANNELS
from hvg.env.chargeworld import Obs


def obs_to_tensors(obs: Obs | list[Obs], device: torch.device | str = "cpu"):
    if isinstance(obs, Obs):
        grid = torch.from_numpy(obs.grid).unsqueeze(0)
        gauges = torch.from_numpy(obs.gauges()).unsqueeze(0)
    else:
        grid = torch.from_numpy(np.stack([o.grid for o in obs], axis=0))
        gauges = torch.from_numpy(np.stack([o.gauges() for o in obs], axis=0))
    return grid.to(device=device, dtype=torch.float32), gauges.to(device=device, dtype=torch.float32)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class GRUActorCritic(nn.Module):
    """Default size lands in the 3–8 million parameter band; smoke uses a tiny net."""

    def __init__(
        self,
        grid_size: int = 16,
        n_channels: int = N_CHANNELS,
        conv_channels: list[int] | None = None,
        hidden_size: int = 256,
        gru_layers: int = 2,
        mlp_size: int = 256,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        conv_channels = list(conv_channels or [16, 32])
        layers: list[nn.Module] = []
        in_c = n_channels
        for out_c in conv_channels:
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        layers += [nn.Flatten()]
        self.conv = nn.Sequential(*layers)
        conv_out = in_c * grid_size * grid_size
        self.encoder = nn.Sequential(
            nn.Linear(conv_out + 2, mlp_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=gru_layers, batch_first=True)
        self.policy = nn.Linear(hidden_size, n_actions)
        self.value = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def initial_hidden(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch, self.hidden_size, device=device)

    def forward(self, grid: torch.Tensor, gauges: torch.Tensor, hidden: torch.Tensor | None = None):
        """grid (B,C,H,W), gauges (B,2); returns logits, value, new_hidden, features."""
        b = grid.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(b, device=grid.device)
        flat = self.conv(grid)
        enc = self.encoder(torch.cat([flat, gauges], dim=-1))
        # GRU expects (B, T, F); we feed a single step.
        out, hidden = self.gru(enc.unsqueeze(1), hidden)
        feat = out.squeeze(1)
        logits = self.policy(feat)
        value = self.value(feat).squeeze(-1)
        return logits, value, hidden, feat

    def act(
        self,
        grid: torch.Tensor,
        gauges: torch.Tensor,
        hidden: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        logits, value, hidden, feat = self.forward(grid, gauges, hidden)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, hidden, feat

    def evaluate_actions(self, grid, gauges, hidden, actions):
        logits, value, hidden, feat = self.forward(grid, gauges, hidden)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value, hidden, feat, logits
