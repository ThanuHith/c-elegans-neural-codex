# C. elegans Neural Worm Simulator

<img width="1837" height="794" alt="image" src="https://github.com/user-attachments/assets/b2f74655-2f03-4050-a16e-fc2e42d96062" />
![Uploading image.png…]()



This project is a modular, research-grade simulation scaffold for a *Caenorhabditis elegans*-inspired digital organism. It combines a spiking neural network, a segmented body model, a 2D environment, and a pharmacology layer so behavior emerges from neural activity instead of from hardcoded movement rules.

## What This Project Is

This repository is a complete simulation platform for a worm-like digital organism inspired by the nervous system of *C. elegans*.

It brings together:

- computational neuroscience
- connectome modeling
- environment sensing
- embodied movement
- drug-response simulation
- desktop visualization
- browser-based exploration
- data logging and analysis

In simple terms, this project simulates a digital worm that senses its world, processes those signals through a neural circuit, moves with a segmented body, and changes behavior under different drug conditions.

## What This Project Can Do

This project can:

- simulate spiking neural activity over time
- propagate signals through weighted synapses
- distinguish sensory, interneuron, and motor roles
- convert motor neuron activity into movement
- generate forward motion, reverse motion, and turning
- detect attractants, repellents, and obstacle-like touch signals
- apply dose-dependent drug perturbations
- compare baseline and drug-exposed behavior
- export logs, plots, and summaries
- run in desktop, headless, and Streamlit web modes

## What You Can Do With This Project

You can use this project to:

- demonstrate how neural circuits produce behavior
- explore connectome-inspired control systems
- compare different drug conditions and doses
- teach computational neuroscience concepts
- build a bioinformatics or AI portfolio project
- run classroom or academic demos
- prototype richer biological datasets
- extend the system toward a more validated research simulator

## Who This Project Is For

This project is useful for:

- students in computational neuroscience
- bioinformatics learners
- AI and ML engineers interested in biologically inspired systems
- simulation developers
- pharmaceutical-science demo builders
- researchers who want an extensible scaffold

## Core Features

### Neural Simulation

The neural engine models:

- membrane potential
- threshold-based firing
- refractory behavior
- delayed synaptic transmission
- excitatory and inhibitory effects
- firing-rate tracking

### Connectome Modeling

The connectome layer provides:

- neuron identity and type
- synapse weights
- synapse signs
- chemical and electrical link types
- graph-based organization
- motor and sensory group extraction

### Embodied Movement

The body engine provides:

- a segmented worm body
- curvature-based locomotion
- forward and reverse drive
- turning from asymmetric neural activity
- collision-aware motion

### Environment Interaction

The environment engine includes:

- attractant gradients
- repellent or toxin-like gradients
- obstacles
- wall interactions
- sensory sampling at the worm head

### Drug Simulation

The pharmacology module supports:

- `baseline`
- `stimulant`
- `sedative`
- `neurotoxin`

These conditions alter:

- excitability
- firing thresholds
- excitation and inhibition balance
- noise
- tonic current
- sensory gain
- motor gain
- conduction strength

### Data Logging And Analysis

The project records:

- neural spikes
- speed
- heading
- trajectory
- sensory drive
- scenario and drug phases

It exports:

- CSV logs
- JSON summaries
- PNG plots

### Multiple Interfaces

The project includes:

- a `pygame` desktop application
- headless batch execution
- a full Streamlit web dashboard

## Biological Foundation

The simulator follows the architecture requested in the project brief:

1. Sensory neurons detect attractants, repellents, and touch-like obstacle cues.
2. Interneurons integrate those signals into forward, reverse, and turning decisions.
3. Motor neurons drive dorsal and ventral body contraction patterns.
4. A segmented body converts motor activity into locomotion.
5. Drug conditions modulate excitability, synaptic strength, sensor gain, and noise.

## What Makes The Behavior Emergent

The behavior is not pre-scripted. Instead:

1. The environment produces signals.
2. Sensory neurons respond to those signals.
3. Interneurons integrate competing cues.
4. Motor neurons shape body curvature and thrust.
5. The body moves.
6. The new body position changes the next sensory input.

This feedback loop creates behavior from system dynamics rather than from hardcoded animation.

## Scientific Note

The included prototype dataset is hand-curated and the 302-neuron mode is a deterministic surrogate expansion designed for software development, teaching, and demonstrations.

To use this as a more biologically grounded system, replace the included CSV inputs with curated published connectome data using the same schema.

## Project Structure

```text
celegans_sim/
  __init__.py
  analysis.py
  body.py
  config.py
  connectome.py
  drugs.py
  environment.py
  logging_utils.py
  neural.py
  simulation.py
  visualization.py
data/
  neurons_prototype.csv
  synapses_prototype.csv
outputs/
run.py
streamlit_app.py
requirements.txt
README.md
PROJECT_DETAILS.md
```

## Architecture Overview

### `celegans_sim/connectome.py`

Loads neuron and synapse data, builds the graph, and organizes sensory and motor groups.

### `celegans_sim/neural.py`

Runs the spiking neural simulation and applies delayed synaptic propagation.

### `celegans_sim/body.py`

Transforms motor activity into body curvature, movement, and turning.

### `celegans_sim/environment.py`

Defines sources, obstacles, and spatial sensing behavior.

### `celegans_sim/drugs.py`

Defines dose-dependent drug profiles.

### `celegans_sim/simulation.py`

Combines all subsystems into one step-based organism simulation.

### `celegans_sim/visualization.py`

Provides the `pygame` desktop interface.

### `streamlit_app.py`

Provides the browser-based Streamlit dashboard.

### `celegans_sim/logging_utils.py`

Handles structured data logging and file export.

### `celegans_sim/analysis.py`

Generates analysis plots from logged metrics.

## Development Phases

### Phase 1: Minimal neural network

Run the default prototype:

```bash
py -3 run.py --connectome prototype
```

### Phase 2: Basic movement simulation

The body model converts motor output into segmented locomotion.

### Phase 3: Surrogate 302-neuron scaling

```bash
py -3 run.py --connectome surrogate302
```

### Phase 4: Environment interaction

The environment includes:

- attractants
- repellents
- obstacles
- gradient sensing

### Phase 5: Drug simulation

Drug conditions modify neural and motor behavior.

### Phase 6: UI and analysis

Desktop and web interfaces expose real-time control and result export.

## How To Run

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the desktop simulator

```bash
py -3 run.py
```

### 4. Run headless experiments

```bash
py -3 run.py --headless --steps 3000
```

### 5. Launch the Streamlit web app

```bash
streamlit run streamlit_app.py
```

## Streamlit Dashboard Features

The Streamlit web app includes:

- play, pause, step, and reset controls
- connectome selection
- scenario selection
- drug and dose controls
- live worm arena rendering
- live neuron heatmap
- speed and activity traces
- trajectory history
- CSV, JSON, and ZIP downloads

## Desktop Controls

Inside the `pygame` app:

- `Space`: pause or resume
- `R`: reset simulation
- `Tab`: cycle environment scenarios
- `1`: baseline
- `2`: stimulant
- `3`: sedative
- `4`: neurotoxin
- `[` and `]`: decrease or increase dose
- `E`: export logs and plots

## Example Experiments

### Compare baseline vs stimulant

Measure:

- speed
- mean activity
- trajectory shape
- directional bias

### Compare sedative vs neurotoxin

Measure:

- locomotion suppression
- reduced firing
- altered response to gradients

### Compare environment scenarios

Use:

- `foraging`
- `obstacle_course`
- `toxin_patch`

### Compare prototype vs scaled mode

Use:

- `prototype`
- `surrogate302`

## Data Outputs

Exports are written under `outputs/` and can include:

- `session_metrics.csv`
- `spike_history.csv`
- `phase_summary.csv`
- `summary.json`
- `trajectory.png`
- `activity_vs_speed.png`

These are useful for:

- reports
- presentations
- GitHub screenshots
- follow-up analysis in notebooks, Excel, or R

## Data Schema

### `neurons_prototype.csv`

Columns:

- `name`
- `neuron_type`
- `subtype`
- `side`
- `baseline_current`
- `threshold`
- `tau_ms`

### `synapses_prototype.csv`

Columns:

- `source`
- `target`
- `weight`
- `sign`
- `delay_ms`
- `kind`

## What This Project Is Good For

This project is especially useful as:

- a capstone project
- a final-year project
- a computational neuroscience demo
- a software portfolio project
- a classroom teaching tool
- a foundation for future research work

## Current Scope And Limitations

This project is strong as a simulation scaffold and educational or demonstration platform, but it is not yet a fully validated biological reproduction of the living worm.

Current limitations:

- the bundled prototype connectome is curated rather than canonical
- the 302-neuron mode is a surrogate scaffold rather than a published ground-truth reconstruction
- the neuron model is simplified
- the body mechanics are abstracted
- the drug effects are functional rather than receptor-kinetic

This makes it strong for:

- learning
- demonstration
- prototyping
- qualitative comparison

It is not yet intended for:

- validated quantitative biological claims
- clinical pharmacology conclusions
- full-fidelity biophysical reproduction

## Research Extension Points

To move toward a more publication-oriented workflow:

1. Replace the prototype and surrogate CSVs with curated connectome edge lists.
2. Add neurotransmitter-specific receptor dynamics.
3. Parameterize neurons using experimentally measured membrane properties.
4. Fit drug-response curves from real assay data.
5. Extend the body to a richer biomechanical model.
6. Replace the neural core with Hodgkin-Huxley, Izhikevich, or NEURON-backed cells.
7. Add validation against known worm behaviors.

## Why This Is A Strong GitHub Project

This repository is a strong GitHub project because it combines:

- neuroscience
- simulation
- AI-inspired behavior
- graph data structures
- visualization
- scientific software design

It is especially good as:

- a portfolio centerpiece
- a multidisciplinary software project
- a research demo
- a teaching showcase

## Suggested GitHub Description

`A biologically inspired C. elegans neural worm simulator with connectome-driven behavior, environmental sensing, drug modulation, desktop visualization, headless analysis, and a Streamlit web dashboard.`

## Suggested GitHub Topics

Possible repository topics:

- `computational-neuroscience`
- `bioinformatics`
- `caenorhabditis-elegans`
- `connectome`
- `python`
- `simulation`
- `streamlit`
- `pygame`
- `scientific-computing`
- `systems-biology`

## More Detailed Documentation

For a longer explanation of what the project does, what it can be used for, and how to present it, see:

- [PROJECT_DETAILS.md](C:/Users/LENOVO/OneDrive/Documents/New%20project/PROJECT_DETAILS.md)

## Beginner-Friendly Walkthrough

If you want to understand the code in order, read the files in this sequence:

1. `run.py`
2. `celegans_sim/config.py`
3. `celegans_sim/connectome.py`
4. `celegans_sim/neural.py`
5. `celegans_sim/environment.py`
6. `celegans_sim/body.py`
7. `celegans_sim/drugs.py`
8. `celegans_sim/simulation.py`
9. `celegans_sim/visualization.py`
10. `celegans_sim/analysis.py`
