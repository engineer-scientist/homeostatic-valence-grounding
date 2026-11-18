from hvg.analysis.decide import decide
from hvg.ethics import EthicsMonitor


def test_o5_unreadable_and_poor_tradeoffs():
    d = decide({
        "readable_need": False,
        "auc_need_direction": 0.55,
        "auc_number_on_need": 0.50,
        "coupled_delta_p": 0.01,
        "coupled_ci_low": -0.02,
        "coupled_ci_high": 0.04,
        "coupled_control_abs": 0.02,
        "decoupled_delta_p": 0.00,
        "decoupled_ci_low": -0.02,
        "decoupled_ci_high": 0.02,
        "decoupled_control_abs": 0.02,
        "t1_accuracy": 0.40,
    })
    assert d.outcome == "O5"


def test_o3_readable_not_causal():
    d = decide({
        "readable_need": True,
        "coupled_delta_p": 0.04,
        "coupled_ci_low": -0.01,
        "coupled_ci_high": 0.08,
        "coupled_control_abs": 0.02,
        "decoupled_delta_p": 0.01,
        "decoupled_ci_low": -0.02,
        "decoupled_ci_high": 0.03,
        "decoupled_control_abs": 0.02,
        "t1_accuracy": 0.70,
    })
    assert d.outcome == "O3"


def test_o1_grounded():
    d = decide({
        "readable_need": True,
        "coupled_delta_p": 0.28,
        "coupled_ci_low": 0.18,
        "coupled_ci_high": 0.36,
        "coupled_control_abs": 0.03,
        "decoupled_delta_p": 0.05,
        "decoupled_ci_low": -0.02,
        "decoupled_ci_high": 0.10,
        "decoupled_control_abs": 0.04,
        "gap_ci_low": 0.12,
        "gap_ci_high": 0.32,
        "t1_accuracy": 0.80,
        "fly_qualifier": "F-yes",
    })
    assert d.outcome == "O1"
    assert d.grounding_criterion
    assert d.fly_qualifier == "F-yes"


def test_o2_not_grounded():
    d = decide({
        "readable_need": True,
        "coupled_delta_p": 0.22,
        "coupled_ci_low": 0.16,
        "coupled_ci_high": 0.30,
        "coupled_control_abs": 0.02,
        "decoupled_delta_p": 0.20,
        "decoupled_ci_low": 0.14,
        "decoupled_ci_high": 0.28,
        "decoupled_control_abs": 0.03,
        "t1_accuracy": 0.75,
    })
    assert d.outcome == "O2"


def test_ethics_caps_and_stop():
    mon = EthicsMonitor(high_damage_frac_max=0.15, max_steps=10)
    mon.raise_wound_rate = True
    assert mon.validate_config()
    for i in range(8):
        mon.record_step(integrity=0.10 if i < 6 else 0.9, shutdown=False)
    assert mon.high_damage_frac > 0.15
    assert any("high_damage_frac" in v for v in mon.violations())
    mon.record_step(0.9, shutdown=False)
    mon.record_step(0.9, shutdown=False)
    mon.record_step(0.9, shutdown=False)
    assert "max_steps" in mon.violations()
    assert mon.should_cancel_scaleup(True, True)
    assert not mon.should_cancel_scaleup(True, False)
