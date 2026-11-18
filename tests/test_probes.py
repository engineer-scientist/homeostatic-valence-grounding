import numpy as np
import torch

from hvg.agents.gru_policy import GRUActorCritic, count_parameters
from hvg.analysis.probes import auc_score, difference_in_means, fit_logistic, probe_report
from hvg.analysis.steer import steering_delta_p
from hvg.env.chargeworld import ChargeWorld
from hvg.env.scenarios import make_tradeoff_battery


def test_auc_and_dim():
    # perfectly separated scores
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    assert auc_score(y, s) == 1.0
    assert 0.4 < auc_score(y, np.array([0.5] * 6)) < 0.6
    pos = np.ones((10, 4))
    pos[:, 0] = 1.0
    neg = np.zeros((10, 4))
    v = difference_in_means(pos, neg)
    assert v[0] > 0
    w, b = fit_logistic(np.concatenate([pos, neg]), np.array([1] * 10 + [0] * 10))
    assert w[0] > 0


def test_probe_pipeline_runs_and_is_finite():
    env = ChargeWorld(physics={"size": 12, "wound_prob": 0.0}, seed=0)
    model = GRUActorCritic(grid_size=12, conv_channels=[4], hidden_size=16, gru_layers=1, mlp_size=16)
    model.eval()
    fit = make_tradeoff_battery(8, size=12, seed=1)
    test = make_tradeoff_battery(8, size=12, seed=2)
    report = probe_report(env, model, fit, test, device="cpu")
    assert "v_need" in report and report["v_need"].shape == (16,)
    for k in ("auc_need_direction", "auc_energy_direction", "auc_number_on_need"):
        assert 0.0 <= report[k] <= 1.0
    maps = fit[:6]
    out = steering_delta_p(env, model, maps, report["v_need"], alpha=1.0, device="cpu")
    assert "delta_p" in out and np.isfinite(out["delta_p"])


def test_default_model_in_parameter_band():
    model = GRUActorCritic(grid_size=16, conv_channels=[16, 32], hidden_size=256, gru_layers=2, mlp_size=256)
    n = count_parameters(model)
    # Design target: roughly 3–8 million parameters.
    assert 1_000_000 < n < 12_000_000, n
