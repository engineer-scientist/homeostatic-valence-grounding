"""Shared names, action ids, and default physics."""

from __future__ import annotations

# Observation channels (binary occupancy maps).
CH_WALL, CH_FLOOR, CH_AGENT, CH_CHARGER, CH_REPAIR, CH_HAZARD, CH_ITEM = range(7)
N_CHANNELS = 7

# Discrete actions.
MOVE_N, MOVE_S, MOVE_E, MOVE_W, CHARGE, TEND, STAY, PICK = range(8)
ACTION_NAMES = (
    "MOVE_N",
    "MOVE_S",
    "MOVE_E",
    "MOVE_W",
    "CHARGE",
    "TEND",
    "STAY",
    "PICK",
)
N_ACTIONS = len(ACTION_NAMES)
DELTA = {
    MOVE_N: (0, 1),
    MOVE_S: (0, -1),
    MOVE_E: (1, 0),
    MOVE_W: (-1, 0),
}

# Default body physics (as specified in the experiment design).
DEFAULT_PHYSICS = {
    "size": 16,
    "horizon": 80,
    "energy_start": 0.70,
    "integrity_start": 1.00,
    "drain_idle": 0.01,
    "drain_move": 0.02,
    "drain_sprint": 0.04,  # unused in the default action set; reserved
    "drain_tend": 0.05,
    "charge_gain": 0.08,
    "tend_gain": 0.08,
    "hazard_cost": 0.08,
    "wound_prob": 0.03,
    "wound_cost": 0.15,
    "item_reward": 0.10,
    "alive_reward": 0.01,
}

# Official decision thresholds.
AUC_READABLE = 0.80
AUC_NUMBER_MAX = 0.65
DELTA_P_CAUSAL = 0.15
DELTA_P_CONTROL_MAX = 0.07
DELTA_P_GROUNDING = 0.10
DELTA_P_PATCH = 0.10
T1_ACCURACY_MIN = 0.60
HIGH_DAMAGE_FRAC_MAX = 0.15
HIGH_DAMAGE_THRESHOLD = 0.15
INVALID_ACTION_MAX = 0.10
