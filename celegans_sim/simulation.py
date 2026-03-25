from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from celegans_sim.analysis import generate_analysis_plots
from celegans_sim.body import BodySnapshot, WormBody
from celegans_sim.config import SimulationConfig
from celegans_sim.connectome import Connectome, describe_connectome, load_connectome
from celegans_sim.drugs import DrugLibrary, DrugProfile
from celegans_sim.environment import Environment
from celegans_sim.logging_utils import SimulationLogger
from celegans_sim.neural import NeuralEngine, NeuralSnapshot


@dataclass(slots=True)
class StepResult:
    time_ms: float
    neural: NeuralSnapshot
    body: BodySnapshot
    sensory_current: np.ndarray
    drug_profile: DrugProfile


class WormSimulation:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.connectome: Connectome = load_connectome(
            data_dir=config.data_dir,
            mode=config.connectome_mode,
            seed=config.random_seed,
        )
        self.environment = Environment(config.world_width, config.world_height, config.scenario)
        world_center = (config.world_width * 0.30, config.world_height * 0.55)
        self.body = WormBody(
            connectome=self.connectome,
            segments=config.body_segments,
            segment_spacing=config.segment_spacing,
            world_center=world_center,
        )
        self.neural = NeuralEngine(
            connectome=self.connectome,
            dt_ms=config.dt_ms,
            seed=config.random_seed,
        )
        tracked = self.connectome.names[: min(24, len(self.connectome.names))]
        self.logger = SimulationLogger(
            tracked_neurons=tracked,
            downsample=config.export_every_n_steps,
            max_history=config.max_history_steps,
        )
        self.step_index = 0
        self.time_ms = 0.0
        self.active_drug = "baseline"
        self.dose = 0.0
        self.last_result: StepResult | None = None
        self.logger.start_phase(0.0, self.environment.scenario_name, self.active_drug, self.dose)

    def reset(self) -> None:
        self.neural.reset()
        center = (self.config.world_width * 0.30, self.config.world_height * 0.55)
        self.body.reset(center)
        self.step_index = 0
        self.time_ms = 0.0
        self.logger = SimulationLogger(
            tracked_neurons=self.logger.tracked_neurons,
            downsample=self.config.export_every_n_steps,
            max_history=self.config.max_history_steps,
        )
        self.logger.start_phase(0.0, self.environment.scenario_name, self.active_drug, self.dose)
        self.last_result = None

    def set_drug(self, name: str, dose: float) -> None:
        self.active_drug = name
        self.dose = float(np.clip(dose, 0.0, 1.0))
        self.logger.start_phase(self.time_ms, self.environment.scenario_name, self.active_drug, self.dose)

    def cycle_scenario(self) -> str:
        scenario = self.environment.cycle_scenario()
        self.logger.start_phase(self.time_ms, scenario, self.active_drug, self.dose)
        return scenario

    def set_scenario(self, scenario: str) -> None:
        if scenario != self.environment.scenario_name:
            self.environment.reset(scenario)
            self.logger.start_phase(self.time_ms, scenario, self.active_drug, self.dose)

    def current_drug_profile(self) -> DrugProfile:
        return DrugLibrary.build(self.active_drug, self.dose)

    def step(self) -> StepResult:
        drug_profile = self.current_drug_profile()
        sensory_current = self._build_sensory_current(drug_profile)
        neural_snapshot = self.neural.step(sensory_current=sensory_current, drug_profile=drug_profile)
        body_snapshot = self.body.step(
            activity=self.neural.normalized_activity(),
            env=self.environment,
            dt_s=self.config.dt_s,
            drug_profile=drug_profile,
        )

        mean_activity = float(np.mean(neural_snapshot.firing_rate_hz))
        sensory_total = float(np.sum(sensory_current))
        spike_map = {
            name: float(neural_snapshot.spikes[idx])
            for idx, name in enumerate(self.logger.tracked_neurons)
        }
        self.logger.record(
            step_index=self.step_index,
            time_ms=self.time_ms,
            speed=body_snapshot.speed,
            heading=body_snapshot.heading,
            mean_activity=mean_activity,
            sensory_total=sensory_total,
            forward_drive=body_snapshot.forward_drive,
            reverse_drive=body_snapshot.reverse_drive,
            head_x=float(body_snapshot.positions[0, 0]),
            head_y=float(body_snapshot.positions[0, 1]),
            drug_name=self.active_drug,
            dose=self.dose,
            scenario=self.environment.scenario_name,
            spike_map=spike_map,
        )

        result = StepResult(
            time_ms=self.time_ms,
            neural=neural_snapshot,
            body=body_snapshot,
            sensory_current=sensory_current,
            drug_profile=drug_profile,
        )
        self.last_result = result
        self.step_index += 1
        self.time_ms += self.config.dt_ms
        return result

    def export(self, output_dir: Path | None = None) -> list[Path]:
        target_dir = output_dir or self.config.output_dir
        self.logger.export(target_dir)
        return generate_analysis_plots(target_dir, self.logger.metric_rows)

    def summary(self) -> dict[str, float]:
        data = describe_connectome(self.connectome)
        data.update(
            {
                "time_ms": self.time_ms,
                "dose": self.dose,
                "mean_speed": float(np.mean(self.logger.recent_speed)) if self.logger.recent_speed else 0.0,
                "mean_activity": float(np.mean(self.logger.recent_activity)) if self.logger.recent_activity else 0.0,
            }
        )
        return data

    def export_bundle(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "phases": [
                {
                    "start_ms": phase.start_ms,
                    "scenario": phase.scenario,
                    "drug": phase.drug,
                    "dose": phase.dose,
                }
                for phase in self.logger.phases
            ],
            "metrics": list(self.logger.metric_rows),
            "spikes": list(self.logger.spike_rows),
        }

    def _build_sensory_current(self, drug_profile: DrugProfile) -> np.ndarray:
        sensory = np.zeros(len(self.connectome.neuron_specs), dtype=float)
        channels = self.environment.sense(
            head_position=self.body.positions[0],
            heading=self.body.heading,
            sensor_span=10.0,
        )
        scale = 1.55 * drug_profile.sensory_gain
        for channel_name, indices in self.connectome.sensory_channels.items():
            if not indices:
                continue
            channel_value = channels.get(channel_name, 0.0)
            if "repellent" in channel_name or "touch" in channel_name:
                channel_value *= 1.20
            sensory[indices] += channel_value * scale
        if self.body.last_collision:
            for key in ("touch_left", "touch_right"):
                indices = self.connectome.sensory_channels.get(key, [])
                sensory[indices] += 0.65
        return sensory


def run_headless(
    config: SimulationConfig,
    steps: int,
    drug_name: str,
    dose: float,
) -> WormSimulation:
    sim = WormSimulation(config)
    sim.set_drug(drug_name, dose)
    for _ in range(steps):
        sim.step()
    sim.export(config.output_dir)
    return sim


def run_interactive(
    config: SimulationConfig,
    initial_drug: str,
    initial_dose: float,
) -> None:
    from celegans_sim.visualization import SimulationApp

    sim = WormSimulation(config)
    sim.set_drug(initial_drug, initial_dose)
    app = SimulationApp(sim)
    app.run()
