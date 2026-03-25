from __future__ import annotations

from dataclasses import dataclass


def _hill(dose: float, ec50: float = 0.45, hill_coeff: float = 1.8) -> float:
    dose = max(0.0, min(1.0, dose))
    if dose == 0.0:
        return 0.0
    numerator = dose**hill_coeff
    denominator = ec50**hill_coeff + numerator
    return numerator / denominator


@dataclass(slots=True)
class DrugProfile:
    name: str = "baseline"
    dose: float = 0.0
    excitability_gain: float = 1.0
    threshold_scale: float = 1.0
    excitation_gain: float = 1.0
    inhibition_gain: float = 1.0
    noise_scale: float = 0.015
    tonic_gain: float = 1.0
    tonic_shift: float = 0.0
    sensory_gain: float = 1.0
    motor_gain: float = 1.0
    conduction_gain: float = 1.0


class DrugLibrary:
    PRESETS = ("baseline", "stimulant", "sedative", "neurotoxin")

    @classmethod
    def build(cls, name: str, dose: float) -> DrugProfile:
        if name == "baseline" or dose <= 0.0:
            return DrugProfile(name="baseline", dose=max(0.0, dose))

        effect = _hill(dose)
        if name == "stimulant":
            return DrugProfile(
                name=name,
                dose=dose,
                excitability_gain=1.0 + 0.45 * effect,
                threshold_scale=1.0 - 0.12 * effect,
                excitation_gain=1.0 + 0.30 * effect,
                inhibition_gain=1.0 - 0.10 * effect,
                noise_scale=0.015 + 0.018 * effect,
                tonic_gain=1.0 + 0.18 * effect,
                sensory_gain=1.0 + 0.22 * effect,
                motor_gain=1.0 + 0.16 * effect,
                conduction_gain=1.0 + 0.08 * effect,
            )
        if name == "sedative":
            return DrugProfile(
                name=name,
                dose=dose,
                excitability_gain=1.0 - 0.28 * effect,
                threshold_scale=1.0 + 0.14 * effect,
                excitation_gain=1.0 - 0.32 * effect,
                inhibition_gain=1.0 + 0.34 * effect,
                noise_scale=0.010,
                tonic_gain=1.0 - 0.10 * effect,
                tonic_shift=-0.05 * effect,
                sensory_gain=1.0 - 0.18 * effect,
                motor_gain=1.0 - 0.30 * effect,
                conduction_gain=1.0 - 0.10 * effect,
            )
        if name == "neurotoxin":
            return DrugProfile(
                name=name,
                dose=dose,
                excitability_gain=1.0 - 0.25 * effect,
                threshold_scale=1.0 + 0.20 * effect,
                excitation_gain=1.0 - 0.45 * effect,
                inhibition_gain=1.0 + 0.08 * effect,
                noise_scale=0.020 + 0.030 * effect,
                tonic_gain=1.0 - 0.12 * effect,
                tonic_shift=-0.08 * effect,
                sensory_gain=1.0 - 0.22 * effect,
                motor_gain=1.0 - 0.42 * effect,
                conduction_gain=1.0 - 0.35 * effect,
            )
        raise ValueError(f"Unsupported drug profile: {name}")
