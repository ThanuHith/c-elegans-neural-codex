from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PhaseRecord:
    start_ms: float
    scenario: str
    drug: str
    dose: float


class SimulationLogger:
    def __init__(self, tracked_neurons: list[str], downsample: int = 4, max_history: int = 2400) -> None:
        self.tracked_neurons = tracked_neurons
        self.downsample = max(1, downsample)
        self.max_history = max_history
        self.metric_rows: list[dict[str, float | str]] = []
        self.spike_rows: list[dict[str, float | str]] = []
        self.phases: list[PhaseRecord] = []
        self.recent_speed: deque[float] = deque(maxlen=max_history)
        self.recent_activity: deque[float] = deque(maxlen=max_history)

    def start_phase(self, time_ms: float, scenario: str, drug: str, dose: float) -> None:
        if self.phases and self.phases[-1].scenario == scenario and self.phases[-1].drug == drug and abs(self.phases[-1].dose - dose) < 1e-6:
            return
        self.phases.append(PhaseRecord(start_ms=time_ms, scenario=scenario, drug=drug, dose=dose))

    def record(
        self,
        step_index: int,
        time_ms: float,
        speed: float,
        heading: float,
        mean_activity: float,
        sensory_total: float,
        forward_drive: float,
        reverse_drive: float,
        head_x: float,
        head_y: float,
        drug_name: str,
        dose: float,
        scenario: str,
        spike_map: dict[str, float],
    ) -> None:
        self.recent_speed.append(speed)
        self.recent_activity.append(mean_activity)
        if step_index % self.downsample != 0:
            return

        self.metric_rows.append(
            {
                "time_ms": time_ms,
                "speed": speed,
                "heading": heading,
                "mean_activity": mean_activity,
                "sensory_total": sensory_total,
                "forward_drive": forward_drive,
                "reverse_drive": reverse_drive,
                "head_x": head_x,
                "head_y": head_y,
                "drug": drug_name,
                "dose": dose,
                "scenario": scenario,
            }
        )

        spike_row: dict[str, float | str] = {"time_ms": time_ms, "drug": drug_name, "scenario": scenario}
        spike_row.update(spike_map)
        self.spike_rows.append(spike_row)

    def export(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(output_dir / "session_metrics.csv", self.metric_rows)
        self._write_csv(output_dir / "spike_history.csv", self.spike_rows)
        self._write_phase_csv(output_dir / "phase_summary.csv")

        mean_speed = float(np.mean(list(self.recent_speed))) if self.recent_speed else 0.0
        mean_activity = float(np.mean(list(self.recent_activity))) if self.recent_activity else 0.0
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mean_speed": mean_speed,
                    "mean_activity": mean_activity,
                    "num_metric_rows": len(self.metric_rows),
                    "num_spike_rows": len(self.spike_rows),
                    "phases": [
                        {
                            "start_ms": phase.start_ms,
                            "scenario": phase.scenario,
                            "drug": phase.drug,
                            "dose": phase.dose,
                        }
                        for phase in self.phases
                    ],
                },
                handle,
                indent=2,
            )
        return output_dir

    def _write_csv(self, path: Path, rows: list[dict[str, float | str]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_phase_csv(self, path: Path) -> None:
        rows = [
            {
                "start_ms": phase.start_ms,
                "scenario": phase.scenario,
                "drug": phase.drug,
                "dose": phase.dose,
            }
            for phase in self.phases
        ]
        self._write_csv(path, rows)
