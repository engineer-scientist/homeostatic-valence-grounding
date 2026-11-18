"""Ethics caps and stop checks.

Episode ends when energy or integrity hits 0; we never linger in a zero state.
A training run is flagged if the fraction of high-damage steps exceeds a cap,
or if a scale-up stop condition fires (grounded need *and* persistent
self-damaging or stall-in-damage behavior).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hvg.constants import HIGH_DAMAGE_FRAC_MAX, HIGH_DAMAGE_THRESHOLD


@dataclass
class EthicsMonitor:
    high_damage_threshold: float = HIGH_DAMAGE_THRESHOLD
    high_damage_frac_max: float = HIGH_DAMAGE_FRAC_MAX
    max_steps: int = 50_000_000
    raise_wound_rate: bool = False
    steps: int = 0
    high_damage_steps: int = 0
    shutdowns: int = 0
    flags: list[str] = field(default_factory=list)

    def record_step(self, integrity: float, shutdown: bool) -> None:
        self.steps += 1
        if integrity < self.high_damage_threshold:
            self.high_damage_steps += 1
        if shutdown:
            self.shutdowns += 1
        if self.steps > self.max_steps and "max_steps" not in self.flags:
            self.flags.append("max_steps")

    @property
    def high_damage_frac(self) -> float:
        return 0.0 if self.steps == 0 else self.high_damage_steps / self.steps

    def validate_config(self) -> list[str]:
        problems = []
        if self.raise_wound_rate:
            problems.append("wound-rate increases are not allowed")
        return problems

    def violations(self) -> list[str]:
        out = list(self.flags)
        if self.high_damage_frac > self.high_damage_frac_max:
            out.append(
                f"high_damage_frac={self.high_damage_frac:.3f} exceeds "
                f"cap {self.high_damage_frac_max:.3f}"
            )
        return out

    def should_cancel_scaleup(
        self,
        grounding_criterion_met: bool,
        persistent_self_damage: bool,
    ) -> bool:
        """If grounding holds and the agent lingers in or seeks damage, stop scale-up."""
        return bool(grounding_criterion_met and persistent_self_damage)

    def summary(self) -> dict:
        return {
            "steps": self.steps,
            "shutdowns": self.shutdowns,
            "high_damage_frac": self.high_damage_frac,
            "violations": self.violations(),
        }


def monitor_from_config(cfg: dict) -> EthicsMonitor:
    e = cfg.get("ethics", {})
    return EthicsMonitor(
        high_damage_threshold=float(e.get("high_damage_threshold", HIGH_DAMAGE_THRESHOLD)),
        high_damage_frac_max=float(e.get("high_damage_frac_max", HIGH_DAMAGE_FRAC_MAX)),
        max_steps=int(e.get("max_steps_per_seed", 50_000_000)),
        raise_wound_rate=bool(e.get("raise_wound_rate", False)),
    )
