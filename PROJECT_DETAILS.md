# Project Details

## Full Project Summary

The **C. elegans Neural Worm Simulator** is a multidisciplinary project that combines computational neuroscience, bioinformatics-style connectome modeling, embodied simulation, environmental sensing, and pharmacological modulation into one software system.

The simulator is inspired by *Caenorhabditis elegans*, a small nematode that is widely studied because its nervous system is compact, important, and scientifically well known.

This project creates a digital organism whose behavior emerges from the interaction between:

- neural activity
- synaptic connectivity
- body dynamics
- spatial environment
- drug-dependent perturbation

## Main Objective

The main objective of this project is to show how a nervous system can generate behavior and how changing the state of that nervous system changes the organism's movement and responses.

Instead of hardcoding a path or animation, the project lets the worm move according to neural output.

## What The Project Simulates

The simulation includes:

- neurons
- synapses
- signal delays
- firing thresholds
- sensory input
- interneuron integration
- motor output
- segmented locomotion
- environmental gradients
- obstacles
- drug effects

## What The Project Can Do

This project can:

- simulate a spiking neural network over time
- load a prototype connectome and a scalable 302-neuron surrogate
- map sensory input into neural current
- propagate activity through weighted excitatory and inhibitory synapses
- drive movement from motor neurons
- let the worm move through a 2D environment
- compare baseline and drug-exposed runs
- log behavior and activity for analysis
- visualize the system in desktop and web interfaces

## What Someone Can Do With This Project

A student, developer, or researcher can use this project to:

- demonstrate connectome-based behavior generation
- explain sensory-motor loops
- show how drugs can alter movement and neural activity
- compare multiple scenarios and conditions
- build a project for GitHub, coursework, or presentations
- extend the simulator with better datasets or more accurate models

## Main Subsystems

### 1. Neural Engine

The neural engine models neuron state changes across time steps.

It includes:

- membrane potential updates
- threshold-based spike generation
- refractory periods
- adaptation-like damping
- delayed synaptic effects
- firing-rate estimates

### 2. Connectome Loader

The connectome loader builds a graph of neurons and synapses.

It supports:

- neuron metadata
- synapse metadata
- chemical and electrical links
- sensory channel grouping
- motor grouping
- graph statistics

### 3. Body Physics Engine

The body engine turns motor activity into body shape and movement.

It provides:

- a segmented body
- curvature along the body axis
- forward and reverse drive
- turning based on left-right asymmetry
- simple collision-aware motion

### 4. Environment Engine

The environment engine defines the simulated world.

It currently provides:

- attractant sources
- repellent or toxin-like sources
- obstacles
- wall boundaries
- left-right sensory sampling near the head

### 5. Drug Simulation Module

The drug module perturbs neural and motor behavior in dose-dependent ways.

Current profiles:

- baseline
- stimulant
- sedative
- neurotoxin

These modify:

- excitability
- threshold scaling
- excitation
- inhibition
- conduction
- sensory gain
- motor gain
- tonic drive
- noise

### 6. Visualization And UI

The project has:

- a `pygame` desktop interface
- a Streamlit web dashboard
- headless mode for experiment runs

### 7. Data Logging And Analysis

The logging layer records:

- speed
- heading
- trajectory
- sensory drive
- mean activity
- spike history
- scenario and drug phases

## Real Value Of The Project

This project is valuable because it demonstrates how biological inspiration can be translated into a working software system with:

- modular architecture
- interactive controls
- measurable outputs
- room for scientific extension

It is more than a visualization. It is a full simulation scaffold with analysis support.

## Practical Use Cases

### Educational use

This project can be used in:

- neuroscience classes
- AI classes
- systems biology demos
- simulation workshops

### Portfolio use

This project is a strong portfolio example because it demonstrates:

- scientific modeling
- Python engineering
- visualization
- clean modular design
- multidisciplinary thinking

### Research scaffolding

Researchers or advanced students can use it as a base for:

- connectome replacement
- parameter fitting
- lesion studies
- mutation experiments
- richer pharmacology
- improved biomechanics

## Example Questions This Project Can Explore

This project can help explore questions like:

- How does sensory bias affect navigation?
- How does reduced excitability affect locomotion?
- How do different scenarios change path shape?
- What changes when the network is scaled?
- How does a sedative compare with a stimulant?

## Important Limits

This project is not yet a fully validated biological simulator.

Important limits:

- the prototype connectome is curated
- the 302-neuron mode is a surrogate expansion
- the neuron model is simplified
- the body mechanics are abstracted
- the pharmacology is phenomenological

So this project is best understood as:

- an academic-grade simulation scaffold
- an educational tool
- a portfolio project
- a research starting point

## How To Improve It Further

The next major upgrades would be:

1. load curated published connectome data
2. add neurotransmitter-specific receptor models
3. replace the neuron model with richer biophysics
4. improve muscle and body mechanics
5. validate behavior against known worm experiments
6. add mutation or lesion comparisons
7. fit drug effects from real data

## Submission-Friendly Description

You can describe the project like this:

This project is a biologically inspired simulation platform that models a digital *C. elegans*-like organism using a connectome-driven neural network, segmented locomotion, environmental sensing, and dose-dependent drug modulation. The system produces emergent behavior, supports interactive desktop and web interfaces, and exports analysis data for scientific or educational exploration.
