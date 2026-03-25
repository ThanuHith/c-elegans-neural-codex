from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SimulationConfig:
    connectome_mode: str = "prototype"
    scenario: str = "foraging"
    output_dir: Path = Path("outputs")
    random_seed: int = 7
    dt_ms: float = 5.0
    world_width: int = 900
    world_height: int = 600
    panel_width: int = 380
    body_segments: int = 20
    segment_spacing: float = 11.0
    body_width: float = 14.0
    target_fps: int = 60
    sim_steps_per_frame: int = 3
    max_history_steps: int = 2400
    export_every_n_steps: int = 4
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")

    @property
    def dt_s(self) -> float:
        return self.dt_ms / 1000.0

    @property
    def window_size(self) -> tuple[int, int]:
        return (self.world_width + self.panel_width, self.world_height)

