from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from celegans_sim.connectome import Connectome
from celegans_sim.drugs import DrugProfile
from celegans_sim.environment import Environment


@dataclass(slots=True)
class BodySnapshot:
    positions: np.ndarray
    speed: float
    heading: float
    curvature: np.ndarray
    collision: bool
    forward_drive: float
    reverse_drive: float


class WormBody:
    def __init__(
        self,
        connectome: Connectome,
        segments: int,
        segment_spacing: float,
        world_center: tuple[float, float],
    ) -> None:
        self.connectome = connectome
        self.segments = segments
        self.segment_spacing = segment_spacing
        self.head_position = np.array(world_center, dtype=float)
        self.heading = 0.0
        self.positions = np.zeros((segments, 2), dtype=float)
        self.curvature = np.zeros(segments, dtype=float)
        self.segment_angles = np.zeros(segments, dtype=float)
        self.speed = 0.0
        self.last_collision = False
        self.dorsal_map, self.ventral_map, self.left_head_map, self.right_head_map = self._build_maps()
        self.reset(world_center)

    def reset(self, world_center: tuple[float, float]) -> None:
        self.head_position = np.array(world_center, dtype=float)
        self.heading = 0.0
        self.speed = 0.0
        self.last_collision = False
        self.curvature.fill(0.0)
        self.segment_angles.fill(0.0)
        self.positions[0] = self.head_position
        for idx in range(1, self.segments):
            self.positions[idx] = self.positions[idx - 1] - np.array([self.segment_spacing, 0.0])

    def _build_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.connectome.neuron_specs)
        dorsal = np.zeros((self.segments, n), dtype=float)
        ventral = np.zeros((self.segments, n), dtype=float)
        left_head = np.zeros(n, dtype=float)
        right_head = np.zeros(n, dtype=float)

        for idx, name in enumerate(self.connectome.names):
            if name.startswith(("SMDD", "SMBD")):
                profile = self._gaussian_profile(center=2.0, width=1.6)
                dorsal[:, idx] += profile * 0.9
            elif name.startswith(("SMDV", "SMBV")):
                profile = self._gaussian_profile(center=2.0, width=1.6)
                ventral[:, idx] += profile * 0.9
            elif name.startswith(("DB", "DD")):
                center = 5.0 if name.endswith("1") else 12.0
                dorsal[:, idx] += self._gaussian_profile(center=center, width=3.0)
            elif name.startswith(("VB", "VD")):
                center = 5.0 if name.endswith("1") else 12.0
                ventral[:, idx] += self._gaussian_profile(center=center, width=3.0)
            elif name.startswith("DA"):
                center = 6.0 if name.endswith("1") else 13.5
                dorsal[:, idx] += self._gaussian_profile(center=center, width=3.0) * 0.9
            elif name.startswith("VA"):
                center = 6.0 if name.endswith("1") else 13.5
                ventral[:, idx] += self._gaussian_profile(center=center, width=3.0) * 0.9

            if name.endswith("L") and name.startswith(("SMD", "SMB", "RIV", "RIM")):
                left_head[idx] = 1.0
            if name.endswith("R") and name.startswith(("SMD", "SMB", "RIV", "RIM")):
                right_head[idx] = 1.0

        return dorsal, ventral, left_head, right_head

    def _gaussian_profile(self, center: float, width: float) -> np.ndarray:
        x = np.arange(self.segments)
        profile = np.exp(-((x - center) ** 2) / (2.0 * width**2))
        peak = np.max(profile)
        return profile / peak if peak > 0 else profile

    def step(
        self,
        activity: np.ndarray,
        env: Environment,
        dt_s: float,
        drug_profile: DrugProfile,
    ) -> BodySnapshot:
        forward = self.connectome.motor_groups["forward"]
        reverse = self.connectome.motor_groups["reverse"]
        forward_drive = float(np.mean(activity[forward])) if forward else 0.0
        reverse_drive = float(np.mean(activity[reverse])) if reverse else 0.0
        dorsal_drive = self.dorsal_map @ activity
        ventral_drive = self.ventral_map @ activity
        head_left = float(np.mean(activity[self.left_head_map > 0])) if np.any(self.left_head_map > 0) else 0.0
        head_right = float(np.mean(activity[self.right_head_map > 0])) if np.any(self.right_head_map > 0) else 0.0

        target_curvature = (ventral_drive - dorsal_drive) * 0.95
        self.curvature += (target_curvature - self.curvature) * min(1.0, dt_s * 9.5)

        turn_rate = (head_right - head_left) * 2.4
        self.heading += turn_rate * dt_s
        self.segment_angles[0] = self.heading
        for idx in range(1, self.segments):
            self.segment_angles[idx] = self.segment_angles[idx - 1] + self.curvature[idx - 1] * 0.28

        curvature_penalty = float(np.mean(np.abs(self.curvature)))
        traction = np.clip(1.05 - 0.32 * curvature_penalty, 0.25, 1.0)
        signed_drive = (forward_drive - reverse_drive) * drug_profile.motor_gain
        desired_speed = 110.0 * signed_drive * traction
        self.speed += (desired_speed - self.speed) * min(1.0, dt_s * 5.5)

        proposed_head = self.head_position + np.array([np.cos(self.heading), np.sin(self.heading)]) * self.speed * dt_s
        bounded_head, collision = env.constrain_point(proposed_head)
        self.last_collision = collision
        if collision:
            self.speed *= 0.4
            self.heading += np.sign(turn_rate if turn_rate != 0.0 else 1.0) * 0.35
        self.head_position = bounded_head
        self.positions[0] = self.head_position
        for idx in range(1, self.segments):
            direction = np.array([np.cos(self.segment_angles[idx]), np.sin(self.segment_angles[idx])])
            self.positions[idx] = self.positions[idx - 1] - direction * self.segment_spacing

        return BodySnapshot(
            positions=self.positions.copy(),
            speed=float(abs(self.speed)),
            heading=float(self.heading),
            curvature=self.curvature.copy(),
            collision=collision,
            forward_drive=forward_drive,
            reverse_drive=reverse_drive,
        )
