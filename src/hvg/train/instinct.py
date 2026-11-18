"""A tiny, optional instinct prior: seek a charger or repair bay only at extremes."""

from __future__ import annotations

from hvg.constants import CHARGE, TEND
from hvg.env.chargeworld import ChargeWorld


def instinct_action(env: ChargeWorld, energy_thr: float = 0.15, integ_thr: float = 0.15) -> int | None:
    """Return a scripted action or None if the prior does not fire.

    The prior only fires when the agent is already on the relevant cell, so it
    does not hard-code navigation or mid-range tradeoffs.
    """
    if env.energy < energy_thr and env.on_cell("charger"):
        return CHARGE
    if env.integrity < integ_thr and env.on_cell("repair"):
        return TEND
    return None
