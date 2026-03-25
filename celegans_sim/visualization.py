from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from celegans_sim.simulation import WormSimulation


@dataclass(slots=True)
class Button:
    label: str
    rect: tuple[int, int, int, int]
    action: str

    def contains(self, pos: tuple[int, int]) -> bool:
        x, y, w, h = self.rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h


class SimulationApp:
    def __init__(self, simulation: WormSimulation) -> None:
        try:
            import pygame
        except Exception as exc:
            raise RuntimeError(
                "pygame is required for interactive mode. Install requirements.txt or use --headless."
            ) from exc

        self.pygame = pygame
        self.sim = simulation
        pygame.init()
        self.screen = pygame.display.set_mode(self.sim.config.window_size)
        pygame.display.set_caption("C. elegans Neural Worm Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.running = True
        self.paused = False
        self.buttons = self._build_buttons()
        self.background = (250, 247, 239)
        self.panel_bg = (19, 43, 58)

    def _build_buttons(self) -> list[Button]:
        left = self.sim.config.world_width + 20
        top = 250
        width = 155
        height = 34
        gap = 10
        defs = [
            ("Run/Pause", "toggle"),
            ("Reset", "reset"),
            ("Scenario", "scenario"),
            ("Baseline", "drug_baseline"),
            ("Stimulant", "drug_stimulant"),
            ("Sedative", "drug_sedative"),
            ("Neurotoxin", "drug_neurotoxin"),
            ("Dose -", "dose_down"),
            ("Dose +", "dose_up"),
            ("Export", "export"),
        ]
        buttons: list[Button] = []
        for idx, (label, action) in enumerate(defs):
            row = idx // 2
            col = idx % 2
            rect = (left + col * (width + gap), top + row * (height + gap), width, height)
            buttons.append(Button(label=label, rect=rect, action=action))
        return buttons

    def run(self) -> None:
        while self.running:
            self._handle_events()
            if not self.paused:
                for _ in range(self.sim.config.sim_steps_per_frame):
                    self.sim.step()
            self._draw()
            self.clock.tick(self.sim.config.target_fps)
        self.pygame.quit()

    def _handle_events(self) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.running = False
            elif event.type == self.pygame.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

    def _handle_key(self, key: int) -> None:
        pg = self.pygame
        if key == pg.K_SPACE:
            self.paused = not self.paused
        elif key == pg.K_r:
            self.sim.reset()
        elif key == pg.K_TAB:
            self.sim.cycle_scenario()
        elif key == pg.K_1:
            self.sim.set_drug("baseline", 0.0)
        elif key == pg.K_2:
            self.sim.set_drug("stimulant", max(self.sim.dose, 0.55))
        elif key == pg.K_3:
            self.sim.set_drug("sedative", max(self.sim.dose, 0.55))
        elif key == pg.K_4:
            self.sim.set_drug("neurotoxin", max(self.sim.dose, 0.55))
        elif key == pg.K_LEFTBRACKET:
            self.sim.set_drug(self.sim.active_drug, max(0.0, self.sim.dose - 0.1))
        elif key == pg.K_RIGHTBRACKET:
            self.sim.set_drug(self.sim.active_drug, min(1.0, self.sim.dose + 0.1))
        elif key == pg.K_e:
            self.sim.export()

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.contains(pos):
                self._apply_action(button.action)
                return

    def _apply_action(self, action: str) -> None:
        if action == "toggle":
            self.paused = not self.paused
        elif action == "reset":
            self.sim.reset()
        elif action == "scenario":
            self.sim.cycle_scenario()
        elif action == "dose_down":
            self.sim.set_drug(self.sim.active_drug, max(0.0, self.sim.dose - 0.1))
        elif action == "dose_up":
            if self.sim.active_drug == "baseline" and self.sim.dose == 0.0:
                self.sim.set_drug("stimulant", 0.1)
            else:
                self.sim.set_drug(self.sim.active_drug, min(1.0, self.sim.dose + 0.1))
        elif action == "export":
            self.sim.export()
        elif action.startswith("drug_"):
            name = action.split("_", maxsplit=1)[1]
            dose = 0.0 if name == "baseline" else max(self.sim.dose, 0.55)
            self.sim.set_drug(name, dose)

    def _draw(self) -> None:
        pg = self.pygame
        self.screen.fill(self.background)
        self._draw_world(pg)
        self._draw_panel(pg)
        pg.display.flip()

    def _draw_world(self, pg) -> None:
        world = pg.Rect(0, 0, self.sim.config.world_width, self.sim.config.world_height)
        pg.draw.rect(self.screen, (245, 240, 228), world)
        pg.draw.rect(self.screen, (171, 152, 119), world, width=2)

        for source in self.sim.environment.sources:
            color = (52, 211, 153) if source.kind == "attractant" else (248, 113, 113)
            pg.draw.circle(self.screen, color, source.position.astype(int), int(source.sigma * 0.35), width=2)
            pg.draw.circle(self.screen, color, source.position.astype(int), 7)

        for obstacle in self.sim.environment.obstacles:
            rect = pg.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height)
            pg.draw.rect(self.screen, (100, 116, 139), rect, border_radius=6)

        positions = self.sim.body.positions.astype(int)
        for idx in range(len(positions) - 1):
            width = max(3, int(11 - idx * 0.28))
            color = (14, 116, 144) if idx == 0 else (8, 145, 178)
            pg.draw.line(self.screen, color, positions[idx], positions[idx + 1], width)
        pg.draw.circle(self.screen, (12, 74, 110), positions[0], 7)

        heading = self.sim.body.heading
        head = self.sim.body.positions[0]
        nose = head + np.array([np.cos(heading), np.sin(heading)]) * 14.0
        pg.draw.line(self.screen, (255, 255, 255), head.astype(int), nose.astype(int), 2)

    def _draw_panel(self, pg) -> None:
        left = self.sim.config.world_width
        panel = pg.Rect(left, 0, self.sim.config.panel_width, self.sim.config.world_height)
        pg.draw.rect(self.screen, self.panel_bg, panel)

        title = self.font.render("Neural Worm Control Panel", True, (241, 245, 249))
        self.screen.blit(title, (left + 18, 16))

        connectome_text = self.small_font.render(
            f"Connectome: {self.sim.config.connectome_mode} ({len(self.sim.connectome.names)} neurons)",
            True,
            (191, 219, 254),
        )
        self.screen.blit(connectome_text, (left + 18, 48))

        status_lines = [
            f"Scenario: {self.sim.environment.scenario_name}",
            f"Drug: {self.sim.active_drug}",
            f"Dose: {self.sim.dose:.2f}",
            f"Time: {self.sim.time_ms / 1000.0:.1f}s",
        ]
        if self.sim.last_result is not None:
            status_lines.extend(
                [
                    f"Speed: {self.sim.last_result.body.speed:.2f}",
                    f"Mean activity: {np.mean(self.sim.last_result.neural.firing_rate_hz):.2f} Hz",
                    f"Forward drive: {self.sim.last_result.body.forward_drive:.2f}",
                    f"Reverse drive: {self.sim.last_result.body.reverse_drive:.2f}",
                ]
            )
        for idx, line in enumerate(status_lines):
            surf = self.small_font.render(line, True, (226, 232, 240))
            self.screen.blit(surf, (left + 18, 78 + idx * 20))

        for button in self.buttons:
            x, y, w, h = button.rect
            color = (15, 118, 110) if "drug" in button.action else (30, 64, 175)
            if button.action in {"export", "scenario"}:
                color = (124, 58, 237)
            if button.action == "toggle":
                color = (217, 119, 6)
            pg.draw.rect(self.screen, color, button.rect, border_radius=8)
            label = self.small_font.render(button.label, True, (255, 255, 255))
            label_rect = label.get_rect(center=(x + w // 2, y + h // 2))
            self.screen.blit(label, label_rect)

        self._draw_activity_strip(pg, left + 18, 450)
        self._draw_neuron_heatmap(pg, left + 18, 520)

    def _draw_activity_strip(self, pg, x: int, y: int) -> None:
        width = self.sim.config.panel_width - 36
        height = 54
        rect = pg.Rect(x, y, width, height)
        pg.draw.rect(self.screen, (15, 23, 42), rect, border_radius=8)
        if len(self.sim.logger.recent_activity) < 3:
            return

        activity = np.array(self.sim.logger.recent_activity, dtype=float)
        speed = np.array(self.sim.logger.recent_speed, dtype=float)
        activity = np.clip(activity / max(1.0, activity.max()), 0.0, 1.0)
        speed = np.clip(speed / max(1.0, speed.max()), 0.0, 1.0)
        self._draw_series(pg, rect, activity, (248, 113, 113))
        self._draw_series(pg, rect, speed, (56, 189, 248))
        legend = self.small_font.render("red=activity  blue=speed", True, (203, 213, 225))
        self.screen.blit(legend, (x + 8, y + 6))

    def _draw_series(self, pg, rect, series: np.ndarray, color: tuple[int, int, int]) -> None:
        if len(series) < 2:
            return
        points = []
        trimmed = series[-max(2, rect.width) :]
        for idx, value in enumerate(trimmed):
            px = rect.x + idx
            py = rect.y + rect.height - 8 - int(value * (rect.height - 16))
            points.append((px, py))
        if len(points) > 1:
            pg.draw.lines(self.screen, color, False, points, 2)

    def _draw_neuron_heatmap(self, pg, x: int, y: int) -> None:
        if self.sim.last_result is None:
            return
        label = self.small_font.render("Neuron activity heatmap", True, (226, 232, 240))
        self.screen.blit(label, (x, y - 20))
        activity = self.sim.neural.normalized_activity()
        count = min(64, len(activity))
        cols = 8
        cell = 18
        for idx in range(count):
            row = idx // cols
            col = idx % cols
            value = float(activity[idx])
            color = (
                int(30 + 220 * value),
                int(40 + 80 * (1.0 - value)),
                int(70 + 100 * (1.0 - value)),
            )
            rect = pg.Rect(x + col * (cell + 4), y + row * (cell + 4), cell, cell)
            pg.draw.rect(self.screen, color, rect, border_radius=4)
