from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass(slots=True)
class NeuronSpec:
    name: str
    neuron_type: str
    subtype: str
    side: str
    baseline_current: float
    threshold: float
    tau_ms: float


@dataclass(slots=True)
class SynapseSpec:
    source: str
    target: str
    weight: float
    sign: int
    delay_ms: float
    kind: str


@dataclass(slots=True)
class Connectome:
    neuron_specs: list[NeuronSpec]
    synapses: list[SynapseSpec]
    graph: nx.DiGraph
    neuron_index: dict[str, int]
    sensory_channels: dict[str, list[int]]
    motor_groups: dict[str, list[int]]
    type_indices: dict[str, np.ndarray]
    outgoing: list[dict[str, np.ndarray]]

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.neuron_specs]

    @property
    def neuron_types(self) -> np.ndarray:
        return np.array([spec.neuron_type for spec in self.neuron_specs], dtype=object)

    @property
    def thresholds(self) -> np.ndarray:
        return np.array([spec.threshold for spec in self.neuron_specs], dtype=float)

    @property
    def tau_ms(self) -> np.ndarray:
        return np.array([spec.tau_ms for spec in self.neuron_specs], dtype=float)

    @property
    def baseline_current(self) -> np.ndarray:
        return np.array([spec.baseline_current for spec in self.neuron_specs], dtype=float)


def load_connectome(data_dir: Path, mode: str, seed: int = 7) -> Connectome:
    if mode == "prototype":
        return _build_from_csv(
            data_dir / "neurons_prototype.csv",
            data_dir / "synapses_prototype.csv",
        )
    if mode == "surrogate302":
        base = _build_from_csv(
            data_dir / "neurons_prototype.csv",
            data_dir / "synapses_prototype.csv",
        )
        return _build_surrogate_302(base, seed=seed)
    raise ValueError(f"Unsupported connectome mode: {mode}")


def _build_from_csv(neuron_path: Path, synapse_path: Path) -> Connectome:
    neuron_specs: list[NeuronSpec] = []
    with neuron_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            neuron_specs.append(
                NeuronSpec(
                    name=row["name"],
                    neuron_type=row["neuron_type"],
                    subtype=row["subtype"],
                    side=row["side"],
                    baseline_current=float(row["baseline_current"]),
                    threshold=float(row["threshold"]),
                    tau_ms=float(row["tau_ms"]),
                )
            )

    synapses: list[SynapseSpec] = []
    with synapse_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            synapses.append(
                SynapseSpec(
                    source=row["source"],
                    target=row["target"],
                    weight=float(row["weight"]),
                    sign=int(row["sign"]),
                    delay_ms=float(row["delay_ms"]),
                    kind=row["kind"],
                )
            )

    return _assemble_connectome(neuron_specs, synapses)


def _assemble_connectome(
    neuron_specs: list[NeuronSpec],
    synapses: list[SynapseSpec],
) -> Connectome:
    graph = nx.DiGraph()
    for spec in neuron_specs:
        graph.add_node(
            spec.name,
            neuron_type=spec.neuron_type,
            subtype=spec.subtype,
            side=spec.side,
        )

    for syn in synapses:
        graph.add_edge(
            syn.source,
            syn.target,
            weight=syn.weight,
            sign=syn.sign,
            delay_ms=syn.delay_ms,
            kind=syn.kind,
        )
        if syn.kind == "electrical" and not graph.has_edge(syn.target, syn.source):
            graph.add_edge(
                syn.target,
                syn.source,
                weight=syn.weight,
                sign=syn.sign,
                delay_ms=syn.delay_ms,
                kind=syn.kind,
            )

    neuron_index = {spec.name: idx for idx, spec in enumerate(neuron_specs)}
    type_indices = _build_type_indices(neuron_specs)
    sensory_channels = _build_sensory_channels(neuron_index)
    motor_groups = _build_motor_groups(neuron_index)
    outgoing = _build_outgoing(neuron_specs, graph, neuron_index)

    return Connectome(
        neuron_specs=neuron_specs,
        synapses=synapses,
        graph=graph,
        neuron_index=neuron_index,
        sensory_channels=sensory_channels,
        motor_groups=motor_groups,
        type_indices=type_indices,
        outgoing=outgoing,
    )


def _build_type_indices(neuron_specs: list[NeuronSpec]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for idx, spec in enumerate(neuron_specs):
        grouped.setdefault(spec.neuron_type, []).append(idx)
    return {key: np.array(value, dtype=int) for key, value in grouped.items()}


def _build_outgoing(
    neuron_specs: list[NeuronSpec],
    graph: nx.DiGraph,
    neuron_index: dict[str, int],
) -> list[dict[str, np.ndarray]]:
    outgoing: list[dict[str, np.ndarray]] = []
    for spec in neuron_specs:
        targets: list[int] = []
        weights: list[float] = []
        signs: list[float] = []
        delays: list[float] = []
        kinds: list[str] = []
        for _, target, data in graph.out_edges(spec.name, data=True):
            targets.append(neuron_index[target])
            weights.append(float(data["weight"]))
            signs.append(float(data["sign"]))
            delays.append(float(data["delay_ms"]))
            kinds.append(str(data["kind"]))
        outgoing.append(
            {
                "targets": np.array(targets, dtype=int),
                "weights": np.array(weights, dtype=float),
                "signs": np.array(signs, dtype=float),
                "delays_ms": np.array(delays, dtype=float),
                "kinds": np.array(kinds, dtype=object),
            }
        )
    return outgoing


def _build_sensory_channels(neuron_index: dict[str, int]) -> dict[str, list[int]]:
    mapping = {
        "attractant_left": ["AWAL", "AWCL", "ASEL"],
        "attractant_right": ["AWAR", "AWCR", "ASER"],
        "repellent_left": ["AWBL", "ASHL", "ADFL"],
        "repellent_right": ["AWBR", "ASHR", "ADFR"],
        "touch_left": ["ASHL", "OLQDL"],
        "touch_right": ["ASHR", "OLQDR"],
    }
    return {
        key: [neuron_index[name] for name in names if name in neuron_index]
        for key, names in mapping.items()
    }


def _build_motor_groups(neuron_index: dict[str, int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "forward": [],
        "reverse": [],
        "dorsal": [],
        "ventral": [],
        "head_left": [],
        "head_right": [],
    }
    for name, idx in neuron_index.items():
        if re.match(r"^(DB|VB)", name):
            groups["forward"].append(idx)
        if re.match(r"^(DA|VA)", name):
            groups["reverse"].append(idx)
        if re.match(r"^(DB|DA|DD|SMDD)", name):
            groups["dorsal"].append(idx)
        if re.match(r"^(VB|VA|VD|SMDV)", name):
            groups["ventral"].append(idx)
        if name.endswith("L"):
            if name.startswith(("SMD", "SMB", "RIV", "RIM")):
                groups["head_left"].append(idx)
        if name.endswith("R"):
            if name.startswith(("SMD", "SMB", "RIV", "RIM")):
                groups["head_right"].append(idx)
    return groups


def _build_surrogate_302(base: Connectome, seed: int) -> Connectome:
    rng = np.random.default_rng(seed)
    neuron_specs = list(base.neuron_specs)
    synapses = list(base.synapses)
    extra_needed = 302 - len(neuron_specs)
    if extra_needed <= 0:
        return base

    template_by_type: dict[str, list[NeuronSpec]] = {}
    for spec in base.neuron_specs:
        template_by_type.setdefault(spec.neuron_type, []).append(spec)

    desired_counts = {"sensory": 90, "interneuron": 112, "motor": 100}
    current_counts = {
        key: sum(1 for spec in neuron_specs if spec.neuron_type == key)
        for key in desired_counts
    }

    ext_specs: list[NeuronSpec] = []
    ext_names: list[str] = []
    for neuron_type, target_count in desired_counts.items():
        template_pool = template_by_type[neuron_type]
        for ordinal in range(current_counts[neuron_type], target_count):
            template = template_pool[ordinal % len(template_pool)]
            name = f"{neuron_type[:3].upper()}_EXT_{ordinal + 1:03d}"
            ext_specs.append(
                NeuronSpec(
                    name=name,
                    neuron_type=neuron_type,
                    subtype=f"{template.subtype}_ext",
                    side="C",
                    baseline_current=template.baseline_current + rng.normal(0.0, 0.04),
                    threshold=template.threshold + rng.normal(0.0, 0.02),
                    tau_ms=max(8.0, template.tau_ms + rng.normal(0.0, 1.0)),
                )
            )
            ext_names.append(name)

    neuron_specs.extend(ext_specs[:extra_needed])
    full_names = [spec.name for spec in neuron_specs]
    hubs = {
        "sensory": [name for name in full_names if name.startswith(("AWA", "AWC", "ASE", "ASH", "AWB"))],
        "interneuron": [name for name in full_names if name.startswith(("AIY", "AIZ", "AIB", "AVA", "AVB", "PVC", "RIM", "RIV"))],
        "motor": [name for name in full_names if name.startswith(("DB", "VB", "DA", "VA", "DD", "VD", "SMD", "SMB"))],
    }

    for spec in neuron_specs[len(base.neuron_specs):]:
        if spec.neuron_type == "sensory":
            targets = rng.choice(hubs["interneuron"], size=3, replace=False)
            for target in targets:
                synapses.append(
                    SynapseSpec(
                        source=spec.name,
                        target=str(target),
                        weight=float(rng.uniform(0.5, 1.1)),
                        sign=1,
                        delay_ms=float(rng.uniform(5.0, 18.0)),
                        kind="chemical",
                    )
                )
        elif spec.neuron_type == "interneuron":
            targets = rng.choice(hubs["interneuron"] + hubs["motor"], size=4, replace=False)
            for target in targets:
                synapses.append(
                    SynapseSpec(
                        source=spec.name,
                        target=str(target),
                        weight=float(rng.uniform(0.35, 1.0)),
                        sign=int(rng.choice([1, 1, 1, -1])),
                        delay_ms=float(rng.uniform(4.0, 15.0)),
                        kind=str(rng.choice(["chemical", "chemical", "electrical"])),
                    )
                )
        else:
            targets = rng.choice(hubs["motor"], size=3, replace=False)
            for target in targets:
                if target == spec.name:
                    continue
                synapses.append(
                    SynapseSpec(
                        source=spec.name,
                        target=str(target),
                        weight=float(rng.uniform(0.2, 0.7)),
                        sign=int(rng.choice([1, -1])),
                        delay_ms=float(rng.uniform(3.0, 10.0)),
                        kind="chemical",
                    )
                )

        feedback_source = str(rng.choice(hubs["interneuron"]))
        synapses.append(
            SynapseSpec(
                source=feedback_source,
                target=spec.name,
                weight=float(rng.uniform(0.2, 0.9)),
                sign=int(rng.choice([1, 1, -1])),
                delay_ms=float(rng.uniform(3.0, 12.0)),
                kind="chemical",
            )
        )

    return _assemble_connectome(neuron_specs, synapses)


def describe_connectome(connectome: Connectome) -> dict[str, float]:
    node_count = len(connectome.neuron_specs)
    edge_count = connectome.graph.number_of_edges()
    density = nx.density(connectome.graph)
    mean_out_degree = edge_count / max(1, node_count)
    return {
        "nodes": float(node_count),
        "edges": float(edge_count),
        "density": float(density),
        "mean_out_degree": float(mean_out_degree),
        "spectral_radius_proxy": float(math.sqrt(max(1.0, edge_count))),
    }

