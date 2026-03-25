from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from celegans_sim.connectome import Connectome
from celegans_sim.drugs import DrugProfile


@dataclass(slots=True)
class NeuralSnapshot:
    membrane_potential: np.ndarray
    firing_rate_hz: np.ndarray
    spikes: np.ndarray
    synaptic_current: np.ndarray


class NeuralEngine:
    def __init__(self, connectome: Connectome, dt_ms: float, seed: int = 7) -> None:
        self.connectome = connectome
        self.dt_ms = dt_ms
        self.n = len(connectome.neuron_specs)
        self.rng = np.random.default_rng(seed)
        self.resting_potential = 0.05
        self.reset_potential = -0.08
        self.refractory_ms = 8.0
        self.current_gain = 4.5
        self.thresholds = connectome.thresholds
        self.tau_ms = connectome.tau_ms
        self.baseline_current = connectome.baseline_current
        self.v = np.full(self.n, self.resting_potential, dtype=float)
        self.firing_rate_hz = np.zeros(self.n, dtype=float)
        self.synaptic_current = np.zeros(self.n, dtype=float)
        self.spikes = np.zeros(self.n, dtype=float)
        self.adaptation = np.zeros(self.n, dtype=float)
        self.refractory_remaining = np.zeros(self.n, dtype=float)
        max_delay_ms = max(
            1.0,
            max(
                float(entry["delays_ms"].max()) if len(entry["delays_ms"]) else 1.0
                for entry in connectome.outgoing
            ),
        )
        self.max_delay_steps = max(
            1,
            int(max_delay_ms / dt_ms) + 2,
        )
        self.delay_buffer = np.zeros((self.max_delay_steps, self.n), dtype=float)
        self.buffer_cursor = 0
        self.rate_alpha = 0.12

    def reset(self) -> None:
        self.v.fill(self.resting_potential)
        self.firing_rate_hz.fill(0.0)
        self.synaptic_current.fill(0.0)
        self.spikes.fill(0.0)
        self.adaptation.fill(0.0)
        self.refractory_remaining.fill(0.0)
        self.delay_buffer.fill(0.0)
        self.buffer_cursor = 0

    def step(
        self,
        sensory_current: np.ndarray,
        drug_profile: DrugProfile,
    ) -> NeuralSnapshot:
        incoming = self.delay_buffer[self.buffer_cursor].copy()
        self.delay_buffer[self.buffer_cursor].fill(0.0)
        self.buffer_cursor = (self.buffer_cursor + 1) % self.max_delay_steps

        noise = self.rng.normal(0.0, drug_profile.noise_scale, size=self.n)
        tonic = self.baseline_current * drug_profile.tonic_gain + drug_profile.tonic_shift
        total_current = tonic + sensory_current + incoming + noise - self.adaptation

        active_mask = self.refractory_remaining <= 0.0
        dv = (
            (self.resting_potential - self.v)
            + total_current * self.current_gain * drug_profile.excitability_gain
        ) / self.tau_ms
        self.v[active_mask] += dv[active_mask] * self.dt_ms
        self.v[~active_mask] = self.reset_potential
        self.refractory_remaining = np.maximum(0.0, self.refractory_remaining - self.dt_ms)
        self.adaptation *= 0.98

        spikes = (self.v >= self.thresholds * drug_profile.threshold_scale) & active_mask
        self.spikes = spikes.astype(float)
        self.v[spikes] = self.reset_potential
        self.refractory_remaining[spikes] = self.refractory_ms
        self.adaptation[spikes] += 0.22
        self.firing_rate_hz = (
            (1.0 - self.rate_alpha) * self.firing_rate_hz
            + self.rate_alpha * self.spikes * (1000.0 / self.dt_ms)
        )
        self.synaptic_current = incoming

        self._schedule_spike_effects(drug_profile)

        return NeuralSnapshot(
            membrane_potential=self.v.copy(),
            firing_rate_hz=self.firing_rate_hz.copy(),
            spikes=self.spikes.copy(),
            synaptic_current=self.synaptic_current.copy(),
        )

    def _schedule_spike_effects(self, drug_profile: DrugProfile) -> None:
        spiking_sources = np.flatnonzero(self.spikes > 0.0)
        for source_idx in spiking_sources:
            bundle = self.connectome.outgoing[source_idx]
            if len(bundle["targets"]) == 0:
                continue

            weights = bundle["weights"].copy()
            signs = bundle["signs"]
            kinds = bundle["kinds"]
            excitation_mask = signs > 0
            inhibition_mask = signs < 0
            weights[excitation_mask] *= drug_profile.excitation_gain
            weights[inhibition_mask] *= drug_profile.inhibition_gain
            weights[kinds == "electrical"] *= drug_profile.conduction_gain

            payload = weights * signs
            delay_steps = np.maximum(
                1,
                np.round(bundle["delays_ms"] / self.dt_ms).astype(int),
            )

            for target_idx, value, delay in zip(bundle["targets"], payload, delay_steps, strict=False):
                slot = (self.buffer_cursor + delay) % self.max_delay_steps
                self.delay_buffer[slot, int(target_idx)] += float(value)

    def normalized_activity(self) -> np.ndarray:
        return np.clip(self.firing_rate_hz / 35.0, 0.0, 1.0)
