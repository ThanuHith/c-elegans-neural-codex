from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ChemicalSource:
    position: np.ndarray
    strength: float
    sigma: float
    kind: str


@dataclass(slots=True)
class Obstacle:
    x: float
    y: float
    width: float
    height: float

    def contains(self, point: np.ndarray) -> bool:
        return (
            self.x <= point[0] <= self.x + self.width
            and self.y <= point[1] <= self.y + self.height
        )

    def distance(self, point: np.ndarray) -> float:
        dx = max(self.x - point[0], 0.0, point[0] - (self.x + self.width))
        dy = max(self.y - point[1], 0.0, point[1] - (self.y + self.height))
        return float(np.hypot(dx, dy))


class Environment:
    def __init__(self, width: int, height: int, scenario: str = "foraging") -> None:
        self.width = width
        self.height = height
        self.margin = 18.0
        self.scenario_name = scenario
        self.sources: list[ChemicalSource] = []
        self.obstacles: list[Obstacle] = []
        self._load_scenario(scenario)

    def reset(self, scenario: str | None = None) -> None:
        if scenario is not None:
            self.scenario_name = scenario
        self.sources.clear()
        self.obstacles.clear()
        self._load_scenario(self.scenario_name)

    def cycle_scenario(self) -> str:
        names = ["foraging", "obstacle_course", "toxin_patch"]
        next_index = (names.index(self.scenario_name) + 1) % len(names)
        self.reset(names[next_index])
        return self.scenario_name

    def _load_scenario(self, scenario: str) -> None:
        if scenario == "foraging":
            self.sources = [
                ChemicalSource(np.array([720.0, 150.0]), 1.20, 120.0, "attractant"),
                ChemicalSource(np.array([250.0, 470.0]), 0.85, 90.0, "repellent"),
            ]
            self.obstacles = [
                Obstacle(360.0, 220.0, 110.0, 35.0),
                Obstacle(520.0, 360.0, 140.0, 35.0),
            ]
        elif scenario == "obstacle_course":
            self.sources = [
                ChemicalSource(np.array([760.0, 120.0]), 1.30, 110.0, "attractant"),
                ChemicalSource(np.array([160.0, 120.0]), 0.70, 80.0, "repellent"),
            ]
            self.obstacles = [
                Obstacle(220.0, 130.0, 50.0, 300.0),
                Obstacle(390.0, 40.0, 50.0, 300.0),
                Obstacle(560.0, 180.0, 50.0, 300.0),
            ]
        elif scenario == "toxin_patch":
            self.sources = [
                ChemicalSource(np.array([730.0, 180.0]), 1.10, 105.0, "attractant"),
                ChemicalSource(np.array([460.0, 300.0]), 1.25, 130.0, "repellent"),
                ChemicalSource(np.array([720.0, 440.0]), 0.75, 90.0, "repellent"),
            ]
            self.obstacles = [
                Obstacle(265.0, 255.0, 90.0, 28.0),
                Obstacle(300.0, 420.0, 180.0, 28.0),
            ]
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def field_strength(self, point: np.ndarray, kind: str) -> float:
        total = 0.0
        for source in self.sources:
            if source.kind != kind:
                continue
            dist_sq = float(np.sum((point - source.position) ** 2))
            total += source.strength * np.exp(-dist_sq / (2.0 * source.sigma**2))
        return total

    def sense(self, head_position: np.ndarray, heading: float, sensor_span: float) -> dict[str, float]:
        forward = np.array([np.cos(heading), np.sin(heading)])
        lateral = np.array([-forward[1], forward[0]])
        nose = head_position + forward * 10.0
        left_sensor = nose + lateral * sensor_span
        right_sensor = nose - lateral * sensor_span

        attract_left = self.field_strength(left_sensor, "attractant")
        attract_right = self.field_strength(right_sensor, "attractant")
        repel_left = self.field_strength(left_sensor, "repellent")
        repel_right = self.field_strength(right_sensor, "repellent")
        touch_left = self._touch_signal(left_sensor)
        touch_right = self._touch_signal(right_sensor)

        return {
            "attractant_left": attract_left,
            "attractant_right": attract_right,
            "repellent_left": repel_left,
            "repellent_right": repel_right,
            "touch_left": touch_left,
            "touch_right": touch_right,
        }

    def constrain_point(self, point: np.ndarray) -> tuple[np.ndarray, bool]:
        bounded = point.astype(float).copy()
        collision = False
        bounded[0] = float(np.clip(bounded[0], self.margin, self.width - self.margin))
        bounded[1] = float(np.clip(bounded[1], self.margin, self.height - self.margin))
        if not np.allclose(bounded, point):
            collision = True

        for obstacle in self.obstacles:
            if obstacle.contains(bounded):
                collision = True
                left_gap = abs(bounded[0] - obstacle.x)
                right_gap = abs(bounded[0] - (obstacle.x + obstacle.width))
                top_gap = abs(bounded[1] - obstacle.y)
                bottom_gap = abs(bounded[1] - (obstacle.y + obstacle.height))
                min_gap = min(left_gap, right_gap, top_gap, bottom_gap)
                if min_gap == left_gap:
                    bounded[0] = obstacle.x - 1.0
                elif min_gap == right_gap:
                    bounded[0] = obstacle.x + obstacle.width + 1.0
                elif min_gap == top_gap:
                    bounded[1] = obstacle.y - 1.0
                else:
                    bounded[1] = obstacle.y + obstacle.height + 1.0

        return bounded, collision

    def _touch_signal(self, point: np.ndarray) -> float:
        wall_distance = min(
            point[0] - self.margin,
            point[1] - self.margin,
            self.width - self.margin - point[0],
            self.height - self.margin - point[1],
        )
        wall_signal = np.exp(-max(0.0, wall_distance) / 12.0)
        obstacle_signal = 0.0
        for obstacle in self.obstacles:
            distance = obstacle.distance(point)
            obstacle_signal = max(obstacle_signal, np.exp(-distance / 10.0))
        return float(max(wall_signal, obstacle_signal))
