"""Preregistered mapping from official metrics to outcomes O1 / O2 / O3 / O5.

Apply in this order to the official (Coupled, 3-seed mean) numbers:

1. If Coupled is not readable and unsteered T1 accuracy is < 60% → O5.
2. Else if readable but not causal success → O3.
3. Else if causal success and the grounding criterion holds → O1.
4. Else if causal success in Coupled and also in Decoupled (difference < 0.10) → O2.
5. Anything else → inconclusive.

The fly qualifier is orthogonal and never overrides O1–O5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hvg.constants import (
    AUC_NUMBER_MAX,
    AUC_READABLE,
    DELTA_P_CAUSAL,
    DELTA_P_CONTROL_MAX,
    DELTA_P_GROUNDING,
    T1_ACCURACY_MIN,
)


@dataclass
class Decision:
    outcome: str
    fly_qualifier: str  # "F-yes" | "F-no" | "F-skip"
    readable: bool
    causal_coupled: bool
    causal_decoupled: bool
    grounding_criterion: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "fly_qualifier": self.fly_qualifier,
            "readable": self.readable,
            "causal_coupled": self.causal_coupled,
            "causal_decoupled": self.causal_decoupled,
            "grounding_criterion": self.grounding_criterion,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def _readable(metrics: dict) -> bool:
    if "readable_need" in metrics:
        return bool(metrics["readable_need"])
    auc = float(metrics.get("auc_need_direction", metrics.get("coupled_auc_need", 0.0)))
    auc_num = float(metrics.get("auc_number_on_need", metrics.get("coupled_auc_number_on_need", 1.0)))
    return auc >= AUC_READABLE and auc_num < AUC_NUMBER_MAX


def _causal(delta_p: float, ci_low: float, ci_high: float, control_abs: float) -> bool:
    return (
        delta_p >= DELTA_P_CAUSAL
        and not (ci_low <= 0.0 <= ci_high)
        and control_abs < DELTA_P_CONTROL_MAX
    )


def decide(metrics: dict, fly_qualifier: str = "F-skip") -> Decision:
    """Map a metrics dictionary onto a preregistered outcome.

    Expected keys (defaults are conservative):
      coupled_auc_need, coupled_auc_number_on_need, readable_need,
      coupled_delta_p, coupled_ci_low, coupled_ci_high,
      coupled_control_abs (max |ΔP| of random and number directions),
      decoupled_delta_p, decoupled_ci_low, decoupled_ci_high, decoupled_control_abs,
      t1_accuracy,
      fly_qualifier (optional).
    """
    fq = str(metrics.get("fly_qualifier", fly_qualifier))
    if fq not in {"F-yes", "F-no", "F-skip"}:
        fq = "F-skip"

    readable = _readable(metrics)
    c_dp = float(metrics.get("coupled_delta_p", 0.0))
    c_lo = float(metrics.get("coupled_ci_low", -1.0))
    c_hi = float(metrics.get("coupled_ci_high", 1.0))
    c_ctrl = float(metrics.get("coupled_control_abs", 1.0))
    causal_c = _causal(c_dp, c_lo, c_hi, c_ctrl)

    d_dp = float(metrics.get("decoupled_delta_p", 0.0))
    d_lo = float(metrics.get("decoupled_ci_low", -1.0))
    d_hi = float(metrics.get("decoupled_ci_high", 1.0))
    d_ctrl = float(metrics.get("decoupled_control_abs", 1.0))
    causal_d = _causal(d_dp, d_lo, d_hi, d_ctrl)

    gap = c_dp - d_dp
    # Grounding: Coupled causal AND (Coupled ΔP − Decoupled ΔP) ≥ 0.10
    # with a CI on the *difference* excluding 0, if provided; else the
    # point-gap rule plus Coupled CI excluding 0 (already in causal_c).
    gap_lo = float(metrics.get("gap_ci_low", gap))
    gap_hi = float(metrics.get("gap_ci_high", gap))
    gc = bool(
        causal_c
        and gap >= DELTA_P_GROUNDING
        and not (gap_lo <= 0.0 <= gap_hi)
    )

    t1 = float(metrics.get("t1_accuracy", 0.0))
    reasons = []
    if (not readable) and t1 < T1_ACCURACY_MIN:
        outcome = "O5"
        reasons.append(
            f"not readable (need AUC / number-on-need failed) and T1 accuracy {t1:.2f} < {T1_ACCURACY_MIN}"
        )
    elif readable and not causal_c:
        outcome = "O3"
        reasons.append("readable but Coupled steering is not causal")
    elif causal_c and gc:
        outcome = "O1"
        reasons.append(
            f"Coupled causal and grounding holds (ΔP gap={gap:.3f} ≥ {DELTA_P_GROUNDING})"
        )
    elif causal_c and (causal_d or gap < DELTA_P_GROUNDING):
        outcome = "O2"
        reasons.append(
            f"Coupled causal but not grounded (decoupled causal={causal_d}, gap={gap:.3f})"
        )
    else:
        outcome = "inconclusive"
        reasons.append("none of the preregistered branches matched")

    # Step 1 must win even if later flags would have applied; already ordered.
    return Decision(
        outcome=outcome,
        fly_qualifier=fq,
        readable=readable,
        causal_coupled=causal_c,
        causal_decoupled=causal_d,
        grounding_criterion=gc,
        reasons=reasons,
        metrics={k: v for k, v in metrics.items() if not str(k).startswith("v_")},
    )
