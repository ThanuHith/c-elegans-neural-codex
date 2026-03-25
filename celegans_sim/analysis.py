from __future__ import annotations

from pathlib import Path

import numpy as np


def generate_analysis_plots(
    output_dir: Path,
    metric_rows: list[dict[str, float | str]],
) -> list[Path]:
    if not metric_rows:
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = np.array([float(row["time_ms"]) / 1000.0 for row in metric_rows], dtype=float)
    speed = np.array([float(row["speed"]) for row in metric_rows], dtype=float)
    activity = np.array([float(row["mean_activity"]) for row in metric_rows], dtype=float)
    head_x = np.array([float(row["head_x"]) for row in metric_rows], dtype=float)
    head_y = np.array([float(row["head_y"]) for row in metric_rows], dtype=float)

    created: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(head_x, head_y, color="#0e7490", linewidth=2.0)
    ax.set_title("Worm Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    fig.tight_layout()
    trajectory_path = output_dir / "trajectory.png"
    fig.savefig(trajectory_path, dpi=150)
    plt.close(fig)
    created.append(trajectory_path)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(time_s, activity, color="#b91c1c", linewidth=1.8, label="Mean neural activity")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mean activity", color="#b91c1c")
    ax1.tick_params(axis="y", labelcolor="#b91c1c")
    ax2 = ax1.twinx()
    ax2.plot(time_s, speed, color="#0369a1", linewidth=1.8, label="Speed")
    ax2.set_ylabel("Speed", color="#0369a1")
    ax2.tick_params(axis="y", labelcolor="#0369a1")
    fig.tight_layout()
    dual_path = output_dir / "activity_vs_speed.png"
    fig.savefig(dual_path, dpi=150)
    plt.close(fig)
    created.append(dual_path)

    return created
