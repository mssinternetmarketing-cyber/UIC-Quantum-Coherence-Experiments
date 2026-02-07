"""
UIC v0.6 — Lambda Sweep with PROPER Quantum/Classical Mixing
==============================================================

ChatGPT-approved implementation:
- Quantum branch: H-CX (entangled Bell state)
- Classical branch: |00⟩ or |11⟩ (separable)
- SAME evolution block for both (CX-RZ-CX stays separable!)
- Post-processing mixing: P_λ = (1-λ)P_Q + λP_C

Both Z-basis and X-basis measurements.
"""

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
)


@dataclass
class NoiseParams:
    idle_t1_us: float = 100.0
    idle_t2_us: float = 80.0
    gate_depol_1q: float = 0.001
    gate_depol_2q: float = 0.01
    readout_error: float = 0.02


def build_noise_model(p: NoiseParams) -> NoiseModel:
    noise = NoiseModel()

    if p.gate_depol_1q > 0:
        err_1q = depolarizing_error(p.gate_depol_1q, 1)
        noise.add_all_qubit_quantum_error(err_1q, ["h", "x", "y", "z", "rz", "sx"])

    if p.gate_depol_2q > 0:
        err_2q = depolarizing_error(p.gate_depol_2q, 2)
        noise.add_all_qubit_quantum_error(err_2q, ["cx", "cz"])

    if p.readout_error > 0:
        ro_err = ReadoutError([
            [1 - p.readout_error, p.readout_error],
            [p.readout_error, 1 - p.readout_error],
        ])
        noise.add_all_qubit_readout_error(ro_err)

    return noise


def _append_idle_relax(qc, qubit, dt_ns: int, t1_us: float, t2_us: float):
    if dt_ns <= 0:
        return
    t_us = dt_ns / 1000.0
    if t1_us > 0:
        p_t1 = 1.0 - math.exp(-t_us / t1_us)
        if p_t1 > 0:
            qc.append(amplitude_damping_error(min(p_t1, 1.0)).to_instruction(), [qubit])
    if t2_us > 0:
        p_t2 = 1.0 - math.exp(-t_us / t2_us)
        if p_t2 > 0:
            qc.append(phase_damping_error(min(p_t2, 1.0)).to_instruction(), [qubit])


def _append_depol_2q(qc, q0, q1, p: float):
    if p > 0:
        qc.append(depolarizing_error(p, 2).to_instruction(), [q0, q1])


def build_quantum_classical_branches(
        delays_us: np.ndarray,
        rounds: int,
        seed_base: int,
        x_basis: bool,
        *,
        idle_t1_us: float,
        idle_t2_us: float,
        depol_p_2q: float,
) -> tuple[List[QuantumCircuit], List[QuantumCircuit]]:
    """
    Build TWO circuit families:
    1. Quantum: H-CX (entangled)
    2. Classical: |00⟩ or |11⟩ (separable)

    Both use SAME evolution block (which stays separable for classical!)
    """
    rng = np.random.default_rng(seed_base)
    quantum_circs = []
    classical_circs = []

    for d_us in delays_us:
        # === QUANTUM BRANCH ===
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc_quantum = QuantumCircuit(q, c)

        # Bell state prep
        qc_quantum.h(q[0])
        qc_quantum.cx(q[0], q[1])
        qc_quantum.barrier()

        # Evolution rounds
        per_round_ns = int(d_us * 1000 / max(rounds, 1)) if d_us > 0 else 0
        for r in range(rounds):
            qc_quantum.cx(q[0], q[1])
            phase = 2 * np.pi * rng.random()
            qc_quantum.rz(phase, q[1])
            qc_quantum.cx(q[0], q[1])
            _append_depol_2q(qc_quantum, q[0], q[1], depol_p_2q)

            if per_round_ns > 0:
                qc_quantum.delay(per_round_ns, q[0], unit="ns")
                qc_quantum.delay(per_round_ns, q[1], unit="ns")
                _append_idle_relax(qc_quantum, q[0], per_round_ns, idle_t1_us, idle_t2_us)
                _append_idle_relax(qc_quantum, q[1], per_round_ns, idle_t1_us, idle_t2_us)

        qc_quantum.barrier()

        # Measurement basis
        if x_basis:
            qc_quantum.h(q[0])
            qc_quantum.h(q[1])
            qc_quantum.barrier()

        qc_quantum.measure(q[0], c[0])
        qc_quantum.measure(q[1], c[1])

        basis_label = "XBASIS" if x_basis else "ZBASIS"
        qc_quantum.metadata = {
            "kind": f"QUANTUM_{basis_label}",
            "delay_us": float(d_us),
            "rounds": rounds,
            "branch": "quantum",
        }
        quantum_circs.append(qc_quantum)

        # === CLASSICAL BRANCH ===
        qc_classical = QuantumCircuit(q, c)

        # Classical prep: |00⟩ or |11⟩ (50/50)
        if rng.random() < 0.5:
            pass  # Start at |00⟩
        else:
            qc_classical.x(q[0])
            qc_classical.x(q[1])
        qc_classical.barrier()

        # SAME evolution block
        per_round_ns = int(d_us * 1000 / max(rounds, 1)) if d_us > 0 else 0
        for r in range(rounds):
            qc_classical.cx(q[0], q[1])
            phase = 2 * np.pi * rng.random()
            qc_classical.rz(phase, q[1])
            qc_classical.cx(q[0], q[1])
            _append_depol_2q(qc_classical, q[0], q[1], depol_p_2q)

            if per_round_ns > 0:
                qc_classical.delay(per_round_ns, q[0], unit="ns")
                qc_classical.delay(per_round_ns, q[1], unit="ns")
                _append_idle_relax(qc_classical, q[0], per_round_ns, idle_t1_us, idle_t2_us)
                _append_idle_relax(qc_classical, q[1], per_round_ns, idle_t1_us, idle_t2_us)

        qc_classical.barrier()

        # SAME measurement basis
        if x_basis:
            qc_classical.h(q[0])
            qc_classical.h(q[1])
            qc_classical.barrier()

        qc_classical.measure(q[0], c[0])
        qc_classical.measure(q[1], c[1])

        qc_classical.metadata = {
            "kind": f"CLASSICAL_{basis_label}",
            "delay_us": float(d_us),
            "rounds": rounds,
            "branch": "classical",
        }
        classical_circs.append(qc_classical)

    return quantum_circs, classical_circs


def run_circuits(
        circuits: List[QuantumCircuit],
        noise_model: NoiseModel,
        shots: int,
        seed: int,
) -> List[Dict]:
    backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    job = backend.run(transpiled, shots=shots)
    result = job.result()

    rows = []
    for i, qc in enumerate(circuits):
        counts = result.get_counts(i)
        meta = qc.metadata or {}

        total = sum(counts.values())
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        p01 = counts.get("01", 0) / total
        p10 = counts.get("10", 0) / total

        rows.append({
            "kind": meta.get("kind", "UNKNOWN"),
            "delay_us": meta.get("delay_us", 0.0),
            "rounds": meta.get("rounds", np.nan),
            "branch": meta.get("branch", ""),
            "shots": shots,
            "counts_json": json.dumps(counts),
            "p00": p00,
            "p11": p11,
            "p01": p01,
            "p10": p10,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_lambda_final", help="output directory")
    ap.add_argument("--shots", type=int, default=2000, help="shots per circuit")
    ap.add_argument("--reps", type=int, default=10, help="repetitions")
    ap.add_argument("--seed", type=int, default=None, help="base seed")
    args = ap.parse_args()

    if args.seed is None:
        args.seed = int(hashlib.sha256(str(random.random()).encode()).hexdigest(), 16) % (2 ** 31)

    os.makedirs(args.out, exist_ok=True)

    noise_params = NoiseParams(
        idle_t1_us=100.0,
        idle_t2_us=80.0,
        gate_depol_1q=0.001,
        gate_depol_2q=0.01,
        readout_error=0.02,
    )
    noise_model = build_noise_model(noise_params)

    delays_us = np.concatenate([
        np.linspace(0, 50, 7),
        np.linspace(60, 200, 8),
    ])

    round_counts = [2, 5, 10]

    all_rows = []

    for rep in range(args.reps):
        print(f"Rep {rep + 1}/{args.reps}...")
        rep_seed = args.seed + rep * 10000

        for rounds in round_counts:
            # Z-basis (computational)
            q_z, c_z = build_quantum_classical_branches(
                delays_us, rounds, rep_seed + rounds * 1000, x_basis=False,
                idle_t1_us=noise_params.idle_t1_us,
                idle_t2_us=noise_params.idle_t2_us,
                depol_p_2q=noise_params.gate_depol_2q,
            )
            all_rows.extend(run_circuits(q_z, noise_model, args.shots, rep_seed + rounds))
            all_rows.extend(run_circuits(c_z, noise_model, args.shots, rep_seed + rounds + 100))

            # X-basis (coherence-sensitive)
            q_x, c_x = build_quantum_classical_branches(
                delays_us, rounds, rep_seed + rounds * 2000, x_basis=True,
                idle_t1_us=noise_params.idle_t1_us,
                idle_t2_us=noise_params.idle_t2_us,
                depol_p_2q=noise_params.gate_depol_2q,
            )
            all_rows.extend(run_circuits(q_x, noise_model, args.shots, rep_seed + rounds + 200))
            all_rows.extend(run_circuits(c_x, noise_model, args.shots, rep_seed + rounds + 300))

        # Add rep column
        for row in all_rows[-(len(delays_us) * len(round_counts) * 4):]:
            row["rep"] = rep

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.out, f"lambda_final_{args.seed}.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✅ Done! Saved to: {out_path}")
    print(f"Total circuits: {len(df)}")
    print(f"\nKind breakdown:")
    print(df["kind"].value_counts())


if __name__ == "__main__":
    main()
