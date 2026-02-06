"""
UIC v0.1 — RPH Simulation Runner (Qiskit 1.x compatible)
========================================================

Runs Aer simulator locally, outputs same CSV format as hardware runner.
Compatible with uic_rph_analysis_v02.py

Requirements:
  - qiskit >= 1.0
  - qiskit-aer
  - numpy

Usage:
  python uic_rph_sim_runner_v01.py --shots 4000 --n 10 --out results_smoketest
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Blinding map (same as hardware)
CONDITION_MAP = {
    "calibration": "Z",
    "isolated": "A",
    "structured_coupling": "B",
    "random_noise": "C",
}


@dataclass
class ExpConfig:
    shots: int
    n_reps: int
    seed: int
    max_delay_us: float
    n_delay_points: int
    t1_max_delay_us: float
    t1_delay_points: int
    feedback_rounds: int
    noise_gates_per_round: int


def _choose_delays_us(max_delay_us: float, n_points: int) -> np.ndarray:
    """Quasi-log spacing; includes 0 and max."""
    if n_points <= 1:
        return np.array([0.0], dtype=float)
    xs = np.geomspace(0.05, 1.0, n_points - 1)
    delays = np.concatenate([[0.0], xs * max_delay_us])
    return delays[:n_points]


def build_t1_circuits(delays_us: np.ndarray) -> List[QuantumCircuit]:
    """Prepare |1>, wait, measure. Fit exponential decay -> T1."""
    circs = []
    for d_us in delays_us:
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        if d_us > 0:
            qc.delay(int(d_us * 1000), 0, unit="ns")
        qc.measure(0, 0)
        qc.metadata = {"kind": "T1", "delay_us": float(d_us), "blind": "Z"}
        circs.append(qc)
    return circs


def build_ramsey_circuits_single(
    delays_us: np.ndarray, blind: str, kind: str = "RAMSEY_SINGLE"
) -> List[QuantumCircuit]:
    """Single-qubit Ramsey: H - delay - H - measure. Visibility decay -> T2*."""
    circs = []
    for d_us in delays_us:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        if d_us > 0:
            qc.delay(int(d_us * 1000), 0, unit="ns")
        qc.h(0)
        qc.measure(0, 0)
        qc.metadata = {"kind": kind, "delay_us": float(d_us), "blind": blind}
        circs.append(qc)
    return circs


def build_bell_ramsey_structured(
    delays_us: np.ndarray, rounds: int
) -> List[QuantumCircuit]:
    """Coherent structured coupling (simulation-friendly)."""
    circs = []
    for d_us in delays_us:
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[1])

        for _ in range(rounds):
            if d_us > 0:
                qc.delay(int(d_us * 1000 / max(rounds, 1)), q[0], unit="ns")
                qc.delay(int(d_us * 1000 / max(rounds, 1)), q[1], unit="ns")
            qc.cx(q[0], q[1])
            qc.rz(math.pi / 8, q[0])
            qc.cx(q[0], q[1])

        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.metadata = {"kind": "BELL_STRUCTURED", "delay_us": float(d_us), "blind": "B"}
        circs.append(qc)
    return circs


def build_bell_ramsey_noise(
    delays_us: np.ndarray, noise_gates_per_round: int, seed: int
) -> List[QuantumCircuit]:
    """Bell prep + delay broken into chunks + random Pauli gates each chunk."""
    rng = random.Random(seed)
    paulis = ["I", "X", "Y", "Z"]
    circs = []

    for d_us in delays_us:
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[1])

        chunks = max(noise_gates_per_round, 1)
        for _ in range(chunks):
            if d_us > 0:
                qc.delay(int(d_us * 1000 / chunks), q[0], unit="ns")
                qc.delay(int(d_us * 1000 / chunks), q[1], unit="ns")

            for qi in [q[0], q[1]]:
                p = rng.choice(paulis)
                if p == "X":
                    qc.x(qi)
                elif p == "Y":
                    qc.y(qi)
                elif p == "Z":
                    qc.z(qi)

        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.metadata = {"kind": "BELL_NOISE", "delay_us": float(d_us), "blind": "C"}
        circs.append(qc)
    return circs


def _mutual_info_from_probs(p: Dict[str, float]) -> float:
    """Calculate mutual information from joint probability distribution."""
    p00 = p.get("00", 0.0)
    p01 = p.get("01", 0.0)
    p10 = p.get("10", 0.0)
    p11 = p.get("11", 0.0)
    px0 = p00 + p01
    px1 = p10 + p11
    py0 = p00 + p10
    py1 = p01 + p11

    def safe_log2(x: float) -> float:
        return math.log(x, 2) if x > 0 else 0.0

    mi = 0.0
    for xy, pxy in [("00", p00), ("01", p01), ("10", p10), ("11", p11)]:
        if pxy <= 0:
            continue
        x = xy[0]
        y = xy[1]
        px = px0 if x == "0" else px1
        py = py0 if y == "0" else py1
        if px > 0 and py > 0:
            mi += pxy * safe_log2(pxy / (px * py))
    return mi


def _shannon_entropy(p: Dict[str, float]) -> float:
    """Calculate Shannon entropy."""
    h = 0.0
    for v in p.values():
        if v > 0:
            h -= v * math.log(v, 2)
    return h


def run_simulation(cfg: ExpConfig, out_dir: str) -> None:
    """Run Aer simulation and output same format as hardware runner."""
    os.makedirs(out_dir, exist_ok=True)

    # Create simulator with noise model
    simulator = AerSimulator(method="density_matrix")

    # Delay schedules
    ramsey_delays = _choose_delays_us(cfg.max_delay_us, cfg.n_delay_points)
    t1_delays = _choose_delays_us(cfg.t1_max_delay_us, cfg.t1_delay_points)

    # Build circuit templates
    t1_circs = build_t1_circuits(t1_delays)
    z_ramsey = build_ramsey_circuits_single(
        ramsey_delays, blind="Z", kind="RAMSEY_SINGLE_CAL"
    )
    a_ramsey = build_ramsey_circuits_single(
        ramsey_delays, blind="A", kind="RAMSEY_SINGLE"
    )
    b_circs = build_bell_ramsey_structured(ramsey_delays, rounds=cfg.feedback_rounds)
    c_circs = build_bell_ramsey_noise(
        ramsey_delays, cfg.noise_gates_per_round, seed=cfg.seed
    )

    # Build all circuits with rep metadata
    all_circs: List[QuantumCircuit] = []
    for rep in range(cfg.n_reps):
        rep_set: List[QuantumCircuit] = []
        rep_set.extend([qc.copy() for qc in t1_circs])
        rep_set.extend([qc.copy() for qc in z_ramsey])
        rep_set.extend([qc.copy() for qc in a_ramsey])
        rep_set.extend([qc.copy() for qc in b_circs])
        rep_set.extend([qc.copy() for qc in c_circs])

        # Tag rep index
        for qc in rep_set:
            qc.metadata = qc.metadata or {}
            qc.metadata["rep"] = rep

        random.Random(cfg.seed + rep).shuffle(rep_set)
        all_circs.extend(rep_set)

    print(f"Running {len(all_circs)} circuits on Aer simulator...")

    # Transpile
    pm = generate_preset_pass_manager(optimization_level=1, backend=simulator)
    transpiled = pm.run(all_circs)

    # Run simulation
    job = simulator.run(transpiled, shots=cfg.shots, seed_simulator=cfg.seed)
    result = job.result()

    # Save decode map
    decode_path = os.path.join(out_dir, "condition_decode_map.json")
    with open(decode_path, "w", encoding="utf-8") as f:
        json.dump({v: k for k, v in CONDITION_MAP.items()}, f, indent=2)

    # Save results CSV
    ts = int(time.time())
    csv_path = os.path.join(out_dir, f"uic_rph_blinded_aer_sim_{ts}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "backend",
                "blind",
                "kind",
                "rep",
                "delay_us",
                "shots",
                "counts_json",
                "p00",
                "p01",
                "p10",
                "p11",
                "mutual_info_bits",
                "shannon_entropy_bits",
                "structured_dynamic_supported",
            ],
        )
        writer.writeheader()

        for i, qc in enumerate(transpiled):
            meta = qc.metadata or {}
            blind = meta.get("blind", "?")
            kind = meta.get("kind", "?")
            delay_us = float(meta.get("delay_us", 0.0))
            rep = int(meta.get("rep", -1))

            counts = result.get_counts(i)
            probs = {k: v / cfg.shots for k, v in counts.items()}

            p00 = probs.get("00", np.nan)
            p01 = probs.get("01", np.nan)
            p10 = probs.get("10", np.nan)
            p11 = probs.get("11", np.nan)

            mi = np.nan
            h = np.nan
            if any(k in probs for k in ["00", "01", "10", "11"]):
                mi = _mutual_info_from_probs(probs)
                h = _shannon_entropy(
                    {k: probs.get(k, 0.0) for k in ["00", "01", "10", "11"]}
                )

            writer.writerow(
                {
                    "timestamp": ts,
                    "backend": "aer_simulator",
                    "blind": blind,
                    "kind": kind,
                    "rep": rep,
                    "delay_us": delay_us,
                    "shots": cfg.shots,
                    "counts_json": json.dumps(counts),
                    "p00": p00,
                    "p01": p01,
                    "p10": p10,
                    "p11": p11,
                    "mutual_info_bits": mi,
                    "shannon_entropy_bits": h,
                    "structured_dynamic_supported": False,  # simulation always uses coherent
                }
            )

    # Save manifest
    manifest = {
        "backend": "aer_simulator",
        "shots": cfg.shots,
        "n_reps": cfg.n_reps,
        "seed": cfg.seed,
        "ramsey_delays_us": ramsey_delays.tolist(),
        "t1_delays_us": t1_delays.tolist(),
        "feedback_rounds": cfg.feedback_rounds,
        "noise_gates_per_round": cfg.noise_gates_per_round,
        "structured_dynamic_supported": False,
        "condition_blind_labels": {
            "calibration": "Z",
            "isolated": "A",
            "structured": "B",
            "noise": "C",
        },
    }

    with open(
        os.path.join(out_dir, f"manifest_aer_sim_{ts}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Saved blinded results to: {csv_path}")
    print(f"✅ Saved decode map to: {decode_path} (DO NOT OPEN UNTIL AFTER STATS)")
    print(f"✅ Saved manifest to: manifest_aer_sim_{ts}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--n", type=int, default=10, dest="n_reps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="uic_sim_out")
    ap.add_argument("--max-delay-us", type=float, default=200.0)
    ap.add_argument("--delay-points", type=int, default=18)
    ap.add_argument("--t1-max-us", type=float, default=300.0)
    ap.add_argument("--t1-points", type=int, default=14)
    ap.add_argument("--feedback-rounds", type=int, default=3)
    ap.add_argument("--noise-gates", type=int, default=6)
    args = ap.parse_args()

    cfg = ExpConfig(
        shots=args.shots,
        n_reps=args.n_reps,
        seed=args.seed,
        max_delay_us=args.max_delay_us,
        n_delay_points=args.delay_points,
        t1_max_delay_us=args.t1_max_us,
        t1_delay_points=args.t1_points,
        feedback_rounds=args.feedback_rounds,
        noise_gates_per_round=args.noise_gates,
    )

    run_simulation(cfg, out_dir=args.out)


if __name__ == "__main__":
    main()
