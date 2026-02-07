"""
UIC v0.1 — RPH Sim Runner v0.4: Clean Lambda-Sweep (Post-Processing Mixture)
=============================================================================

What's new vs v0.3:
- Removed broken BELL_DETERMINISTIC circuit builder
- Now generates:
  1) Pure quantum Bell circuits (BELL_STRUCTURED evolution)
  2) Classical reference probabilities (stored in metadata)
- Lambda mixing happens in POST-PROCESSING (analysis script)

This is epistemically cleaner and physically correct.
"""

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
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

    # 1q depolarizing
    if p.gate_depol_1q > 0:
        err_1q = depolarizing_error(p.gate_depol_1q, 1)
        noise.add_all_qubit_quantum_error(err_1q, ["h", "x", "y", "z", "rz", "sx"])

    # 2q depolarizing
    if p.gate_depol_2q > 0:
        err_2q = depolarizing_error(p.gate_depol_2q, 2)
        noise.add_all_qubit_quantum_error(err_2q, ["cx", "cz"])

    # Readout error (simple bitflip)
    if p.readout_error > 0:
        from qiskit_aer.noise import ReadoutError

        ro_err = ReadoutError(
            [
                [1 - p.readout_error, p.readout_error],
                [p.readout_error, 1 - p.readout_error],
            ]
        )
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


def _append_depol_1q(qc, q, p: float):
    if p > 0:
        qc.append(depolarizing_error(p, 1).to_instruction(), [q])


def build_t1_circuits(delays_us: np.ndarray) -> List[QuantumCircuit]:
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
    delays_us: np.ndarray,
    blind: str,
    kind: str = "RAMSEY_SINGLE",
    *,
    idle_t1_us: float,
    idle_t2_us: float,
) -> List[QuantumCircuit]:
    circs = []
    for d_us in delays_us:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        if d_us > 0:
            dt_ns = int(d_us * 1000)
            qc.delay(dt_ns, 0, unit="ns")
            _append_idle_relax(qc, 0, dt_ns, idle_t1_us, idle_t2_us)
        qc.h(0)
        qc.measure(0, 0)
        qc.metadata = {"kind": kind, "delay_us": float(d_us), "blind": blind}
        circs.append(qc)
    return circs


def build_bell_ramsey_structured(
    delays_us: np.ndarray,
    rounds: int,
    blind: str,
    *,
    idle_t1_us: float,
    idle_t2_us: float,
    depol_p_2q: float,
) -> List[QuantumCircuit]:
    circs = []
    for d_us in delays_us:
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[1])

        per_round_ns = int(d_us * 1000 / max(rounds, 1)) if d_us > 0 else 0

        # ONLY run evolution rounds if delay > 0
        if d_us > 0:
            for _ in range(rounds):
                if per_round_ns > 0:
                    qc.delay(per_round_ns, q[0], unit="ns")
                    qc.delay(per_round_ns, q[1], unit="ns")
                    _append_idle_relax(qc, q[0], per_round_ns, idle_t1_us, idle_t2_us)
                    _append_idle_relax(qc, q[1], per_round_ns, idle_t1_us, idle_t2_us)
                qc.cx(q[0], q[1])
                qc.rz(math.pi / 8, q[0])
                qc.cx(q[0], q[1])
                _append_depol_2q(qc, q[0], q[1], depol_p_2q)

        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.metadata = {
            "kind": "BELL_STRUCTURED",
            "delay_us": float(d_us),
            "blind": blind,
        }
        circs.append(qc)
    return circs


def build_bell_ramsey_noise(
    delays_us: np.ndarray,
    noise_gates_per_round: int,
    seed: int,
    blind: str,
    *,
    idle_t1_us: float,
    idle_t2_us: float,
    depol_p_1q: float,
) -> List[QuantumCircuit]:
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
        per_chunk_ns = int(d_us * 1000 / chunks) if d_us > 0 else 0
        for _ in range(chunks):
            if per_chunk_ns > 0:
                qc.delay(per_chunk_ns, q[0], unit="ns")
                qc.delay(per_chunk_ns, q[1], unit="ns")
                _append_idle_relax(qc, q[0], per_chunk_ns, idle_t1_us, idle_t2_us)
                _append_idle_relax(qc, q[1], per_chunk_ns, idle_t1_us, idle_t2_us)
            for qi in [q[0], q[1]]:
                p = rng.choice(paulis)
                if p == "X":
                    qc.x(qi)
                elif p == "Y":
                    qc.y(qi)
                elif p == "Z":
                    qc.z(qi)
                _append_depol_1q(qc, qi, depol_p_1q)

        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.metadata = {"kind": "BELL_NOISE", "delay_us": float(d_us), "blind": blind}
        circs.append(qc)
    return circs


def generate_classical_probabilities(seed: int) -> Dict[str, float]:
    """
    Generate deterministic classical probabilities for a Bell-pair measurement.

    In a local hidden variable model, the outcome is predetermined.
    We'll use a simple model: hidden variable determines if both qubits
    are correlated (00 or 11) or anti-correlated (01 or 10).

    For simplicity: 50/50 split between 00 and 11 (classical correlation).
    """
    rng = random.Random(seed)
    # Classical model: always correlated outcomes
    if rng.random() < 0.5:
        return {"00": 1.0, "11": 0.0, "01": 0.0, "10": 0.0}
    else:
        return {"00": 0.0, "11": 1.0, "01": 0.0, "10": 0.0}


def build_lambda_sweep_circuits(
    delays_us: np.ndarray,
    lambda_values: List[float],
    rounds: int,
    seed_base: int,
    *,
    idle_t1_us: float,
    idle_t2_us: float,
    depol_p_2q: float,
) -> List[QuantumCircuit]:
    """
    Build circuits for lambda-sweep.

    We'll generate PURE QUANTUM circuits (BELL_STRUCTURED),
    but store classical reference probabilities in metadata for post-processing.
    """
    circs = []
    for lam in lambda_values:
        blind_token = f"L{int(lam * 10):02d}"

        for d_us in delays_us:
            q = QuantumRegister(2, "q")
            c = ClassicalRegister(2, "c")
            qc = QuantumCircuit(q, c)
            qc.h(q[0])
            qc.cx(q[0], q[1])

            # Pure quantum evolution
            per_round_ns = int(d_us * 1000 / max(rounds, 1)) if d_us > 0 else 0

            # ONLY run evolution rounds if delay > 0
            if d_us > 0:
                for _ in range(rounds):
                    if per_round_ns > 0:
                        qc.delay(per_round_ns, q[0], unit="ns")
                        qc.delay(per_round_ns, q[1], unit="ns")
                        _append_idle_relax(qc, q[0], per_round_ns, idle_t1_us, idle_t2_us)
                        _append_idle_relax(qc, q[1], per_round_ns, idle_t1_us, idle_t2_us)
                    qc.cx(q[0], q[1])
                    qc.rz(math.pi / 8, q[0])
                    qc.cx(q[0], q[1])
                    _append_depol_2q(qc, q[0], q[1], depol_p_2q)

            qc.measure(q[0], c[0])
            qc.measure(q[1], c[1])

            # Store classical reference for post-processing
            classical_probs = generate_classical_probabilities(
                seed_base + int(lam * 1000) + int(d_us)
            )

            qc.metadata = {
                "kind": "BELL_LAMBDA_SWEEP",
                "delay_us": float(d_us),
                "blind": blind_token,
                "lambda": float(lam),
                "classical_ref": json.dumps(classical_probs),
            }
            circs.append(qc)

    return circs


def build_lambda_xbasis_circuits(
        delays_us: List[float],
        lambda_values: List[float],
        rounds: int,
        seed_base: int,
        idle_t1_us: float,
        idle_t2_us: float,
        depol_p_2q: float,
) -> List[QuantumCircuit]:
    """
    X-BASIS Lambda sweep: H⊗H before measurement to probe coherence

    This rotates to the X-basis where quantum/classical states are distinguishable!
    Expected: Parity(t, λ) = (1-λ) * e^(-t/T2*)
    """
    circuits = []
    rng = np.random.default_rng(seed_base)

    for lam in lambda_values:
        for delay in delays_us:
            qc = QuantumCircuit(2, 2)

            # Bell state preparation
            qc.h(0)
            qc.cx(0, 1)
            qc.barrier()

            # Evolution rounds (same as Z-basis)
            for r in range(rounds):
                qc.cx(0, 1)
                phase_offset = 2 * np.pi * rng.random()
                qc.rz(phase_offset, 1)
                qc.cx(0, 1)

                if delay > 0:
                    sub_delay = delay / rounds
                    qc.delay(int(sub_delay * 1000), 0, unit="ns")
                    qc.delay(int(sub_delay * 1000), 1, unit="ns")

            qc.barrier()

            # 🔥 KEY DIFFERENCE: Rotate to X-basis before measurement!
            qc.h(0)
            qc.h(1)
            qc.barrier()

            qc.measure([0, 1], [0, 1])

            qc.metadata = {
                "kind": "BELL_LAMBDA_XBASIS",
                "lambda": float(lam),
                "delay_us": float(delay),
                "rounds": rounds,
                "classical_ref": "",
                "blind": f"xbasis_lam{lam:.2f}_d{delay:.1f}_r{rounds}",
            }
            circuits.append(qc)

    return circuits


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

        # Normalize to probabilities
        total = sum(counts.values())
        p00 = (
            counts.get("00", 0) / total
            if "00" in counts or "11" in counts
            else counts.get("0", 0) / total
        )
        p11 = (
            counts.get("11", 0) / total
            if "00" in counts or "11" in counts
            else counts.get("1", 0) / total
        )
        p01 = counts.get("01", 0) / total
        p10 = counts.get("10", 0) / total

        rows.append(
            {
                "kind": meta.get("kind", "UNKNOWN"),
                "blind": meta.get("blind", ""),
                "delay_us": meta.get("delay_us", 0.0),
                "lambda": meta.get("lambda", np.nan),
                "classical_ref": meta.get("classical_ref", ""),
                "rounds": meta.get("rounds", np.nan),
                "shots": shots,
                "counts_json": json.dumps(counts),
                "p00": p00,
                "p11": p11,
                "p01": p01,
                "p10": p10,
            }
        )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="results_lambda_sweep_clean", help="output directory"
    )
    ap.add_argument("--shots", type=int, default=2000, help="shots per circuit")
    ap.add_argument("--reps", type=int, default=10, help="repetitions per condition")
    ap.add_argument("--seed", type=int, default=None, help="base random seed")
    args = ap.parse_args()

    if args.seed is None:
        args.seed = int(
            hashlib.sha256(str(random.random()).encode()).hexdigest(), 16
        ) % (2**31)

    os.makedirs(args.out, exist_ok=True)

    # Noise params
    noise_params = NoiseParams(
        idle_t1_us=100.0,
        idle_t2_us=80.0,
        gate_depol_1q=0.001,
        gate_depol_2q=0.01,
        readout_error=0.02,
    )
    noise_model = build_noise_model(noise_params)

    # Delay schedule
    delays_us = np.concatenate(
        [
            np.linspace(0, 50, 7),
            np.linspace(60, 200, 8),
        ]
    )

    # Lambda values
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_rows = []

    for rep in range(args.reps):
        print(f"Rep {rep+1}/{args.reps}...")
        rep_seed = args.seed + rep * 10000

        # T1 calibration
        t1_circs = build_t1_circuits(delays_us)
        all_rows.extend(run_circuits(t1_circs, noise_model, args.shots, rep_seed))

        # RAMSEY baseline (A)
        ramsey_circs = build_ramsey_circuits_single(
            delays_us,
            blind="A",
            kind="RAMSEY_SINGLE",
            idle_t1_us=noise_params.idle_t1_us,
            idle_t2_us=noise_params.idle_t2_us,
        )
        all_rows.extend(
            run_circuits(ramsey_circs, noise_model, args.shots, rep_seed + 1)
        )

        # BELL_STRUCTURED (B)
        bell_struct_circs = build_bell_ramsey_structured(
            delays_us,
            rounds=10,
            blind="B",
            idle_t1_us=noise_params.idle_t1_us,
            idle_t2_us=noise_params.idle_t2_us,
            depol_p_2q=noise_params.gate_depol_2q,
        )
        all_rows.extend(
            run_circuits(bell_struct_circs, noise_model, args.shots, rep_seed + 2)
        )

        # BELL_NOISE (C)
        bell_noise_circs = build_bell_ramsey_noise(
            delays_us,
            noise_gates_per_round=10,
            seed=rep_seed + 3,
            blind="C",
            idle_t1_us=noise_params.idle_t1_us,
            idle_t2_us=noise_params.idle_t2_us,
            depol_p_1q=noise_params.gate_depol_1q,
        )
        all_rows.extend(
            run_circuits(bell_noise_circs, noise_model, args.shots, rep_seed + 4)
        )

        # Lambda sweep with multiple round counts (Z-basis)
        round_counts = [0, 2, 5, 10]
        for rounds in round_counts:
            lambda_circs = build_lambda_sweep_circuits(
                delays_us,
                lambda_values,
                rounds=rounds,
                seed_base=rep_seed + 5 + rounds * 1000,
                idle_t1_us=noise_params.idle_t1_us,
                idle_t2_us=noise_params.idle_t2_us,
                depol_p_2q=noise_params.gate_depol_2q,
            )
            # Add rounds metadata
            for circ in lambda_circs:
                circ.metadata["rounds"] = rounds

            all_rows.extend(
                run_circuits(lambda_circs, noise_model, args.shots, rep_seed + 6 + rounds)
            )

        # X-basis Lambda sweep (NEW!)
        for rounds in round_counts:
            xbasis_circs = build_lambda_xbasis_circuits(
                delays_us,
                lambda_values,
                rounds=rounds,
                seed_base=rep_seed + 1000 + rounds * 1000,
                idle_t1_us=noise_params.idle_t1_us,
                idle_t2_us=noise_params.idle_t2_us,
                depol_p_2q=noise_params.gate_depol_2q,
            )
            # Add rounds metadata
            for circ in xbasis_circs:
                circ.metadata["rounds"] = rounds

            all_rows.extend(
                run_circuits(xbasis_circs, noise_model, args.shots, rep_seed + 2000 + rounds)
            )

        # Count how many lambda circuits we just added (all rounds combined)
        total_lambda_circs = len(delays_us) * len(lambda_values) * len(round_counts) * 2  # Z + X basis
        for row in all_rows[-total_lambda_circs:]:
            row["rep"] = rep


    # Save
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.out, f"lambda_sweep_clean_{args.seed}.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✅ Done! Saved to: {out_path}")
    print(f"Total circuits: {len(df)}")
    print(f"\nKind breakdown:")
    print(df["kind"].value_counts())


if __name__ == "__main__":
    main()
