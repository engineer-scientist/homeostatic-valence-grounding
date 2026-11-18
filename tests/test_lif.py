from pathlib import Path

from hvg.fly.lif import load_connectome, run_stimulus_protocol, shuffle_wiring

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_connectome.json"


def test_sugar_drives_mn9_more_than_shuffle():
    conn = load_connectome(FIXTURE)
    assert "sugar_0" in conn.names
    sugar = run_stimulus_protocol(conn, "sugar", "MN9", duration_ms=250, shuffle=False, seed=1, dt=0.2)
    shuf = run_stimulus_protocol(conn, "sugar", "MN9", duration_ms=250, shuffle=True, seed=1, dt=0.2)
    # The intact sugar pathway should drive the feeding motor neuron.
    assert sugar["rate_hz"] > 20.0
    # Shuffling destinations should weaken that specific pathway.
    assert shuf["rate_hz"] < sugar["rate_hz"] - 5.0


def test_silence_blocks_and_shuffle_preserves_counts():
    conn = load_connectome(FIXTURE)
    silenced = run_stimulus_protocol(
        conn, "sugar", "MN9", duration_ms=200, silence_group="sugar", shuffle=False, seed=0, dt=0.2
    )
    assert silenced["rate_hz"] < 5.0
    sh = shuffle_wiring(conn, seed=0)
    assert len(sh.weight) == len(conn.weight)
    assert abs(float(sh.weight.sum()) - float(conn.weight.sum())) < 1e-9
    assert sorted(sh.weight.tolist()) == sorted(conn.weight.tolist())
