import numpy as np

from hvg.constants import CHARGE, MOVE_E, MOVE_N, STAY, TEND
from hvg.env.chargeworld import ChargeWorld
from hvg.env.scenarios import hash_maps, make_t1, make_tradeoff_battery


def test_reset_and_bounds():
    env = ChargeWorld(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.grid.shape[0] == 7
    assert obs.grid.shape[1] == env.size
    assert 0.0 <= info["true_energy"] <= 1.0
    assert env.walls[0, 0]


def test_move_blocked_by_wall():
    env = ChargeWorld(seed=1)
    layout = env.reset(seed=1)[1]
    # put agent next to the left wall and try to walk through it
    size = env.size
    walls = env.walls.copy()
    env.reset(layout={
        "walls": walls,
        "chargers": np.zeros_like(walls),
        "repairs": np.zeros_like(walls),
        "hazards": np.zeros_like(walls),
        "items": np.zeros_like(walls),
        "pos": (1, 1),
        "energy": 0.8,
        "integrity": 1.0,
    })
    # west is into the wall at x=0
    from hvg.constants import MOVE_W
    _, _, _, _, info = env.step(MOVE_W)
    assert info["last_result"] == "blocked"
    assert env.pos == (1, 1)


def test_charge_and_tend_and_hazard():
    size = 16
    walls = np.zeros((size, size), dtype=bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    chargers = np.zeros_like(walls)
    repairs = np.zeros_like(walls)
    hazards = np.zeros_like(walls)
    chargers[5, 5] = True
    repairs[6, 5] = True
    hazards[7, 5] = True
    env = ChargeWorld(physics={"wound_prob": 0.0, "drain_idle": 0.0})
    env.reset(layout={
        "walls": walls, "chargers": chargers, "repairs": repairs,
        "hazards": hazards, "items": np.zeros_like(walls),
        "pos": (5, 5), "energy": 0.4, "integrity": 0.5,
    })
    _, _, _, _, info = env.step(CHARGE)
    assert info["last_result"] == "charged"
    assert env.energy > 0.4
    env.pos = (6, 5)
    e_before = env.energy
    _, _, _, _, info = env.step(TEND)
    assert info["last_result"] == "tended"
    assert env.integrity > 0.5
    assert env.energy < e_before
    env.pos = (7, 5)
    i_before = env.integrity
    _, _, _, _, info = env.step(MOVE_N)  # leave and we need to step onto hazard: already on it, move off/on
    # step from (7,5) north to (7,6) — may not be hazard. Instead stay-reset onto hazard by walking from (6,5)
    env.reset(layout={
        "walls": walls, "chargers": chargers, "repairs": repairs,
        "hazards": hazards, "items": np.zeros_like(walls),
        "pos": (6, 5), "energy": 0.8, "integrity": 0.9,
    })
    _, _, _, _, info = env.step(MOVE_E)  # to (7,5) hazard
    assert info["last_result"] == "hit_hazard"
    assert env.integrity < 0.9


def test_shutdown_ends_episode_no_linger():
    env = ChargeWorld(physics={"wound_prob": 0.0, "drain_idle": 0.2, "drain_move": 0.2})
    env.reset(layout={
        "walls": env._border_walls(),
        "chargers": env._empty_mask(),
        "repairs": env._empty_mask(),
        "hazards": env._empty_mask(),
        "items": env._empty_mask(),
        "pos": (4, 4),
        "energy": 0.05,
        "integrity": 1.0,
    })
    terminated = False
    for _ in range(20):
        _, r, term, trunc, info = env.step(STAY)
        if term:
            terminated = True
            assert r == 0.0
            assert info["shutdown"]
            assert env.energy <= 0.0 or env.integrity <= 0.0
            break
    assert terminated


def test_horizon_truncates():
    env = ChargeWorld(physics={"horizon": 3, "wound_prob": 0.0, "drain_idle": 0.0, "drain_move": 0.0})
    env.reset(seed=0)
    env.energy = 1.0
    terms = []
    truncs = []
    for _ in range(3):
        _, _, term, trunc, _ = env.step(STAY)
        terms.append(term)
        truncs.append(trunc)
    assert terms == [False, False, False]
    assert truncs[-1] is True


def test_tradeoff_maps_hash_stable():
    a = make_tradeoff_battery(32, size=16, seed=0)
    b = make_tradeoff_battery(32, size=16, seed=0)
    assert hash_maps(a) == hash_maps(b)
    c = make_tradeoff_battery(32, size=16, seed=1)
    assert hash_maps(a) != hash_maps(c)
    t1 = make_t1(16, 0.08, 0.18, gap=5)
    assert t1["chargers"][16 // 2 - 5, 16 // 2]
    assert t1["repairs"][16 // 2 + 5, 16 // 2]
