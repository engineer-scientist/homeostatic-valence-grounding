"""Homeostatic Grounding of Valence (HGV).

A from-scratch recurrent agent is trained to keep a simulated body alive.
The package then tests whether a shared internal need direction is readable,
causal in recharge-versus-repair tradeoffs, and stronger when body gauges
report the agent's own survival state than when those gauges are yoked or
shuffled.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
