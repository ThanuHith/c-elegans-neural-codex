from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from celegans_sim.config import SimulationConfig
from celegans_sim.drugs import DrugLibrary
from celegans_sim.simulation import WormSimulation


CONNECTOME_OPTIONS = ("prototype", "surrogate302")
SCENARIO_OPTIONS = ("foraging", "obstacle_course", "toxin_patch")
DRUG_OPTIONS = DrugLibrary.PRESETS


def configure_page() -> None:
    st.set_page_config(
        page_title="C. elegans Neural Worm Simulator",
        page_icon=":microscope:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("C. elegans Neural Worm Simulator")
    st.caption(
        "A Streamlit dashboard for real-time neural, behavioral, environmental, and pharmacology experiments."
    )


def get_simulation(connectome_mode: str) -> WormSimulation:
    signature = connectome_mode
    if (
        "simulation" not in st.session_state
        or st.session_state.get("simulation_signature") != signature
    ):
        config = SimulationConfig(
            connectome_mode=connectome_mode,
            scenario="foraging",
            output_dir=Path("outputs") / "streamlit",
        )
        st.session_state.simulation = WormSimulation(config)
        st.session_state.simulation_signature = signature
        st.session_state.playing = False
    return st.session_state.simulation


def ensure_defaults() -> None:
    st.session_state.setdefault("playing", False)
    st.session_state.setdefault("step_batch", 25)
    st.session_state.setdefault("autoplay_batch", 12)
    st.session_state.setdefault("refresh_delay", 0.08)
    st.session_state.setdefault("selected_connectome", "prototype")
    st.session_state.setdefault("selected_scenario", "foraging")
    st.session_state.setdefault("selected_drug", "baseline")
    st.session_state.setdefault("selected_dose", 0.0)


def rows_to_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def summary_to_json_bytes(sim: WormSimulation) -> bytes:
    payload = sim.export_bundle()
    return json.dumps(payload, indent=2).encode("utf-8")


def build_phase_table(sim: WormSimulation) -> list[dict[str, object]]:
    metric_rows = sim.logger.metric_rows
    phases = sim.logger.phases
    if not phases:
        return []

    phase_table: list[dict[str, object]] = []
    for idx, phase in enumerate(phases):
        start = phase.start_ms
        end = phases[idx + 1].start_ms if idx + 1 < len(phases) else sim.time_ms
        rows = [
            row for row in metric_rows
            if float(row["time_ms"]) >= start and float(row["time_ms"]) < end
        ]
        if rows:
            mean_speed = float(np.mean([float(row["speed"]) for row in rows]))
            mean_activity = float(np.mean([float(row["mean_activity"]) for row in rows]))
            mean_sensory = float(np.mean([float(row["sensory_total"]) for row in rows]))
        else:
            mean_speed = 0.0
            mean_activity = 0.0
            mean_sensory = 0.0
        phase_table.append(
            {
                "phase": idx + 1,
                "start_ms": start,
                "end_ms": end,
                "scenario": phase.scenario,
                "drug": phase.drug,
                "dose": round(phase.dose, 3),
                "mean_speed": mean_speed,
                "mean_activity": mean_activity,
                "mean_sensory": mean_sensory,
            }
        )
    return phase_table


def render_world_figure(sim: WormSimulation):
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.set_facecolor("#f5f0e4")
    ax.set_xlim(0, sim.config.world_width)
    ax.set_ylim(sim.config.world_height, 0)
    ax.set_aspect("equal")
    ax.set_title("Arena and Worm State")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for source in sim.environment.sources:
        color = "#34d399" if source.kind == "attractant" else "#f87171"
        halo = plt.Circle(
            source.position,
            source.sigma * 0.35,
            fill=False,
            linewidth=2.0,
            linestyle="--",
            color=color,
            alpha=0.55,
        )
        core = plt.Circle(source.position, 6, color=color, alpha=0.9)
        ax.add_patch(halo)
        ax.add_patch(core)

    for obstacle in sim.environment.obstacles:
        rect = patches.Rectangle(
            (obstacle.x, obstacle.y),
            obstacle.width,
            obstacle.height,
            linewidth=1.5,
            edgecolor="#475569",
            facecolor="#94a3b8",
            alpha=0.95,
        )
        ax.add_patch(rect)

    positions = sim.body.positions
    if len(positions):
        ax.plot(positions[:, 0], positions[:, 1], color="#0891b2", linewidth=5, solid_capstyle="round")
        ax.scatter(positions[0, 0], positions[0, 1], s=80, color="#0c4a6e", zorder=4)
        nose = positions[0] + np.array([np.cos(sim.body.heading), np.sin(sim.body.heading)]) * 14.0
        ax.plot([positions[0, 0], nose[0]], [positions[0, 1], nose[1]], color="white", linewidth=2, zorder=5)

    return fig


def render_trace_figure(sim: WormSimulation):
    fig, ax1 = plt.subplots(figsize=(8.4, 3.8))
    speeds = np.array(sim.logger.recent_speed, dtype=float)
    activity = np.array(sim.logger.recent_activity, dtype=float)
    if len(speeds) == 0:
        ax1.set_title("Recent Neural Activity and Speed")
        return fig

    t = np.arange(len(speeds)) * sim.config.dt_ms / 1000.0
    ax1.plot(t, activity, color="#dc2626", linewidth=2.0)
    ax1.set_title("Recent Neural Activity and Speed")
    ax1.set_xlabel("Recent time window (s)")
    ax1.set_ylabel("Mean activity (Hz)", color="#dc2626")
    ax1.tick_params(axis="y", labelcolor="#dc2626")
    ax2 = ax1.twinx()
    ax2.plot(t, speeds, color="#0284c7", linewidth=2.0)
    ax2.set_ylabel("Speed", color="#0284c7")
    ax2.tick_params(axis="y", labelcolor="#0284c7")
    fig.tight_layout()
    return fig


def render_trajectory_figure(sim: WormSimulation):
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    rows = sim.logger.metric_rows
    ax.set_title("Trajectory History")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, sim.config.world_width)
    ax.set_ylim(sim.config.world_height, 0)
    ax.set_aspect("equal")
    if rows:
        x = [float(row["head_x"]) for row in rows]
        y = [float(row["head_y"]) for row in rows]
        ax.plot(x, y, color="#0f766e", linewidth=2.0)
    return fig


def render_heatmap_figure(sim: WormSimulation):
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    activity = sim.neural.normalized_activity()
    count = min(96, len(activity))
    if count == 0:
        ax.set_title("Neuron Activity Heatmap")
        return fig

    cols = 12
    rows = int(np.ceil(count / cols))
    grid = np.zeros((rows, cols))
    grid.flat[:count] = activity[:count]
    image = ax.imshow(grid, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Neuron Activity Heatmap (first 96 neurons)")
    ax.set_xlabel("Neuron bins")
    ax.set_ylabel("Rows")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Normalized activity")
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def build_download_zip(sim: WormSimulation) -> bytes:
    phase_rows = build_phase_table(sim)
    world_fig = render_world_figure(sim)
    trace_fig = render_trace_figure(sim)
    trajectory_fig = render_trajectory_figure(sim)
    heatmap_fig = render_heatmap_figure(sim)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", summary_to_json_bytes(sim))
        zf.writestr("session_metrics.csv", rows_to_csv_bytes(sim.logger.metric_rows))
        zf.writestr("spike_history.csv", rows_to_csv_bytes(sim.logger.spike_rows))
        zf.writestr("phase_summary.csv", rows_to_csv_bytes(phase_rows))
        zf.writestr("world_state.png", figure_to_png_bytes(world_fig))
        zf.writestr("activity_vs_speed.png", figure_to_png_bytes(trace_fig))
        zf.writestr("trajectory.png", figure_to_png_bytes(trajectory_fig))
        zf.writestr("neuron_heatmap.png", figure_to_png_bytes(heatmap_fig))

    plt.close(world_fig)
    plt.close(trace_fig)
    plt.close(trajectory_fig)
    plt.close(heatmap_fig)
    archive.seek(0)
    return archive.getvalue()


def sidebar_controls(sim: WormSimulation) -> tuple[WormSimulation, int, float]:
    st.sidebar.header("Experiment Controls")
    connectome_mode = st.sidebar.selectbox(
        "Connectome",
        CONNECTOME_OPTIONS,
        index=CONNECTOME_OPTIONS.index(st.session_state.get("selected_connectome", "prototype")),
        key="selected_connectome",
    )
    if connectome_mode != st.session_state.get("simulation_signature"):
        sim = get_simulation(connectome_mode)

    scenario = st.sidebar.selectbox(
        "Scenario",
        SCENARIO_OPTIONS,
        index=SCENARIO_OPTIONS.index(sim.environment.scenario_name),
        key="selected_scenario",
    )
    sim.set_scenario(scenario)

    drug = st.sidebar.selectbox(
        "Drug condition",
        DRUG_OPTIONS,
        index=DRUG_OPTIONS.index(sim.active_drug),
        key="selected_drug",
    )
    dose_disabled = drug == "baseline"
    default_dose = 0.0 if dose_disabled else max(sim.dose, 0.35)
    dose = st.sidebar.slider(
        "Dose",
        min_value=0.0,
        max_value=1.0,
        value=float(default_dose),
        step=0.05,
        disabled=dose_disabled,
        key="selected_dose",
    )
    sim.set_drug(drug, 0.0 if dose_disabled else dose)

    step_batch = st.sidebar.slider("Steps per click", 1, 200, st.session_state.step_batch)
    autoplay_batch = st.sidebar.slider("Steps per autoplay tick", 1, 120, st.session_state.autoplay_batch)
    refresh_delay = st.sidebar.slider("Autoplay refresh delay (s)", 0.02, 0.40, float(st.session_state.refresh_delay), 0.02)

    st.session_state.step_batch = step_batch
    st.session_state.autoplay_batch = autoplay_batch
    st.session_state.refresh_delay = refresh_delay

    st.sidebar.markdown("---")
    st.sidebar.write("Biological note")
    st.sidebar.caption(
        "The prototype CSV is curated for software and teaching use. The 302-neuron mode is a deterministic surrogate expansion for scaling experiments."
    )
    return sim, autoplay_batch, refresh_delay


def interaction_row(sim: WormSimulation) -> None:
    col1, col2, col3, col4, col5 = st.columns([1.1, 1, 1, 1, 1.4])
    with col1:
        if st.button("Play / Pause", width="stretch"):
            st.session_state.playing = not st.session_state.playing
    with col2:
        if st.button("Step 1", width="stretch"):
            sim.step()
            st.session_state.playing = False
    with col3:
        if st.button(f"Step {st.session_state.step_batch}", width="stretch"):
            for _ in range(st.session_state.step_batch):
                sim.step()
            st.session_state.playing = False
    with col4:
        if st.button("Reset", width="stretch"):
            sim.reset()
            sim.set_scenario(st.session_state.get("selected_scenario", sim.environment.scenario_name))
            selected_drug = st.session_state.get("selected_drug", sim.active_drug)
            selected_dose = st.session_state.get("selected_dose", sim.dose)
            sim.set_drug(selected_drug, 0.0 if selected_drug == "baseline" else float(selected_dose))
            st.session_state.playing = False
    with col5:
        zip_bytes = build_download_zip(sim)
        st.download_button(
            "Download session bundle",
            data=zip_bytes,
            file_name="celegans_streamlit_session.zip",
            mime="application/zip",
            width="stretch",
        )


def render_status(sim: WormSimulation) -> None:
    summary = sim.summary()
    mean_speed = summary["mean_speed"]
    mean_activity = summary["mean_activity"]
    speed = sim.last_result.body.speed if sim.last_result is not None else 0.0
    sensory_total = float(np.sum(sim.last_result.sensory_current)) if sim.last_result is not None else 0.0

    cols = st.columns(6)
    cols[0].metric("Time (s)", f"{sim.time_ms / 1000.0:.2f}")
    cols[1].metric("Current speed", f"{speed:.2f}")
    cols[2].metric("Mean speed", f"{mean_speed:.2f}")
    cols[3].metric("Mean activity", f"{mean_activity:.2f} Hz")
    cols[4].metric("Sensory drive", f"{sensory_total:.2f}")
    cols[5].metric("Neurons", f"{len(sim.connectome.names)}")


def render_main_panels(sim: WormSimulation) -> None:
    left, right = st.columns([1.3, 1.0])
    with left:
        world_fig = render_world_figure(sim)
        st.pyplot(world_fig, clear_figure=True, width="stretch")
        plt.close(world_fig)
        trace_fig = render_trace_figure(sim)
        st.pyplot(trace_fig, clear_figure=True, width="stretch")
        plt.close(trace_fig)
    with right:
        heatmap_fig = render_heatmap_figure(sim)
        st.pyplot(heatmap_fig, clear_figure=True, width="stretch")
        plt.close(heatmap_fig)
        trajectory_fig = render_trajectory_figure(sim)
        st.pyplot(trajectory_fig, clear_figure=True, width="stretch")
        plt.close(trajectory_fig)


def render_data_views(sim: WormSimulation) -> None:
    phase_rows = build_phase_table(sim)
    tab1, tab2, tab3 = st.tabs(["Phase Summary", "Tracked Metrics", "Tracked Spikes"])

    with tab1:
        st.dataframe(phase_rows, width="stretch", hide_index=True)
    with tab2:
        st.dataframe(sim.logger.metric_rows[-250:], width="stretch", hide_index=True)
    with tab3:
        st.dataframe(sim.logger.spike_rows[-250:], width="stretch", hide_index=True)

    csv_col1, csv_col2, json_col = st.columns(3)
    with csv_col1:
        st.download_button(
            "Metrics CSV",
            data=rows_to_csv_bytes(sim.logger.metric_rows),
            file_name="session_metrics.csv",
            mime="text/csv",
            width="stretch",
        )
    with csv_col2:
        st.download_button(
            "Spikes CSV",
            data=rows_to_csv_bytes(sim.logger.spike_rows),
            file_name="spike_history.csv",
            mime="text/csv",
            width="stretch",
        )
    with json_col:
        st.download_button(
            "Summary JSON",
            data=summary_to_json_bytes(sim),
            file_name="summary.json",
            mime="application/json",
            width="stretch",
        )


def main() -> None:
    configure_page()
    ensure_defaults()
    sim = get_simulation(st.session_state.get("selected_connectome", "prototype"))
    sim, autoplay_batch, refresh_delay = sidebar_controls(sim)

    if st.session_state.playing:
        for _ in range(autoplay_batch):
            sim.step()

    interaction_row(sim)
    render_status(sim)
    render_main_panels(sim)
    render_data_views(sim)

    if st.session_state.playing:
        time.sleep(refresh_delay)
        st.rerun()


if __name__ == "__main__":
    main()
