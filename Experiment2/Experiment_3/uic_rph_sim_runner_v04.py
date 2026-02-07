
***

## FILE 3: `uic_rph_sim_runner_v04.py`

This is long, so I'll give you the COMPLETE working code:

```python
"""
UIC v0.4 — Deterministic λ-Sweep Runner
========================================

Changes from v0.3:
- DETERMINISTIC λ implementation (no per-round stochastic branching)
- Reduced to 4 feedback rounds (from 8) to minimize noise accumulation
- Same noise model, delays, analysis pipeline

For each λ:
  n_structured = int(λ * 4)
  n_noise = 4 - n_structured
  Apply n_structured blocks, then n_noise blocks (deterministic order)

Usage:
  python uic_rph_sim_runner_v04.py --shots 4000 --n 10 --lambda-steps 11 --out results_exp3_deterministic
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import thermal_relaxation_error, depolarizing_error
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


CONDITION_MAP_ABCZ = {
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
    
    idle_t1_us: float
    idle_t2_us: float
    
    depol_p_1q: float
    depol_p_2q: float


def _choose_delays_us(max_delay_us: float, n_points: int) -> np.ndarray:
    if n_points <= 2:
        return np.array([0.0, float(max_delay_us)])
    dense = np.linspace(0.0, min(40.0, max_delay_us), max(6, n_points // 2))
    sparse = np.linspace(min(40.0, max_delay_us), float(max_delay_us), n_points - len(dense))
    delays = np.unique(np.concatenate([dense, sparse]))
    return delays


def _append_idle_relax(qc: QuantumCircuit, qubit, dt_ns: int, t1_us: float, t2_us: float) -> None:
    if dt_ns <= 0:
        return
    t1_ns = float(t1_us) * 1000.0
    t2_ns = float(t2_us) * 1000.0
    err = thermal_relaxation_error(t1_ns, t2_ns, float(dt_ns))
    qc.append(err.to_instruction(), [qubit])


def _append_depol_1q(qc: QuantumCircuit, qubit, p: float) -> None:
    if p <= 0:
        return
    err = depolarizing_error(p, 1)
    qc.append(err.to_instruction(), [qubit])


def _append_depol_2q(qc: QuantumCircuit, q0, q1, p: float) -> None:
    if p <= 0:
        return
    err = depolarizing_error(p, 2)
    qc.append(err.to_instruction(), [q0, q1])


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
        qc.metadata = {"kind": "BELL_STRUCTURED", "delay_us": float(d_us), "blind": "B"}
        circs.append(qc)
    return circs


def build_bell_ramsey_noise(
    delays_us: np.ndarray,
    noise_gates_per_round: int,
    seed: int,
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
        qc.metadata = {"kind": "BELL_NOISE", "delay_us": float(d_us), "blind": "C"}
        circs.append(qc)
    return circs


def build_bell_ramsey_deterministic(
    delays_us: np.ndarray,
    total_rounds: int,
    lam: float,
    seed: int,
    *,
    idle_t1_us: float,
    idle_t2_us: float,
    depol_p_2q: float,
    depol_p_1q: float,
) -> List[QuantumCircuit]:
    """
    DETERMINISTIC λ implementation.
    
    For each circuit:
      - n_structured = int(λ * total_rounds)
      - n_noise = total_rounds - n_structured
      - Apply n_structured structured blocks
      - Apply n_noise noise blocks
      
    No stochastic branching per round!
    """
    rng = random.Random(seed + int(lam * 1e6))
    paulis = ["I", "X", "Y", "Z"]
    lam = float(max(0.0, min(1.0, lam)))
    
    n_structured = int(lam * total_rounds)
    n_noise = total_rounds - n_structured
    
    circs = []
    for d_us in delays_us:
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[1])
        
        per_round_ns = int(d_us * 1000 / max(total_rounds, 1)) if d_us > 0 else 0
        
        # Apply structured blocks
        for _ in range(n_structured):
            if per_round_ns > 0:
                qc.delay(per_round_ns, q[0], unit="ns")
                qc.delay(per_round_ns, q[1], unit="ns")
                _append_idle_relax(qc, q[0], per_round_ns, idle_t1_us, idle_t2_us)
                _append_idle_relax(qc, q[1], per_round_ns, idle_t1_us, idle_t2_us)
            
            qc.cx(q[0], q[1])
            qc.rz(math.pi / 8, q[0])
            qc.cx(q[0], q[1])
            _append_depol_2q(qc, q[0], q[1], depol_p_2q)
        
        # Apply noise blocks
        for _ in range(n_noise):
            if per_round_ns > 0:
                qc.delay(per_round_ns, q[0], unit="ns")
                qc.delay(per_round_ns, q[1], unit="ns")
                _append_idle_relax(qc, q[0], per_round_ns, idle_t1_us, idle_t2_us)
                _append_idle_relax(qc, q[1], per_round_ns, idle_t1_us, idle_t2_us)
            
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
        qc.metadata = {
            "kind": "BELL_DETERMINISTIC",
            "delay_us": float(d_us),
            "lambda": float(lam),
        }
        circs.append(qc)
    return circs


def _counts_to_probs(counts: Dict[str, int], shots: int) -> Dict[str, float]:
    return {k: v / shots for k, v in counts.items()}


def _mutual_info_from_probs(p: Dict[str, float]) -> float:
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
    for (xy, pxy) in [("00", p00), ("01", p01), ("10", p10), ("11", p11)]:
        if pxy <= 0:
            continue
        x = xy[0]
        y = xy[1]
        px = px0 if x == "0" else px1
        py = py0 if y == "0" else py1
        mi += pxy * (safe_log2(pxy) - safe_log2(px) - safe_log2(py))
    return mi


def _shannon_entropy(p: Dict[str, float]) -> float:
    h = 0.0
    for v in p.values():
        if v > 0:
            h -= v * math.log(v, 2)
    return h


def _make_lambda_blind_map(lams: List[float]) -> Tuple[Dict[str, str], Dict[str, float]]:
    tokens = [f"L{idx:02d}" for idx in range(len(lams))]
    rng = random.Random(1337)
    rng.shuffle(tokens)
    
    blind_map = {}
    lam_map = {}
    for tok, lam in zip(tokens, lams):
        blind_map[tok] = f"LAMBDA_{lam:.3f}"
        lam_map[tok] = float(lam)
    return blind_map, lam_map


def run_simulation(cfg: ExpConfig, out_dir: str, lambda_steps: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    simulator = AerSimulator(method="density_matrix")
    
    ramsey_delays = _choose_delays_us(cfg.max_delay_us, cfg.n_delay_points)
    t1_delays = _choose_delays_us(cfg.t1_max_delay_us, cfg.t1_delay_points)
    
    t1_circs = build_t1_circuits(t1_delays)
    z_ramsey = build_ramsey_circuits_single(
        ramsey_delays,
        blind="Z",
        kind="RAMSEY_SINGLE_CAL",
        idle_t1_us=cfg.idle_t1_us,
        idle_t2_us=cfg.idle_t2_us,
    )
    a_ramsey = build_ramsey_circuits_single(
        ramsey_delays,
        blind="A",
        kind="RAMSEY_SINGLE",
        idle_t1_us=cfg.idle_t1_us,
        idle_t2_us=cfg.idle_t2_us,
    )
    b_circs = build_bell_ramsey_structured(
        ramsey_delays,
        rounds=cfg.feedback_rounds,
        idle_t1_us=cfg.idle_t1_us,
        idle_t2_us=cfg.idle_t2_us,
        depol_p_2q=cfg.depol_p_2q,
    )
    c_circs = build_bell_ramsey_noise(
        ramsey_delays,
        cfg.noise_gates_per_round,
        seed=cfg.seed,
        idle_t1_us=cfg.idle_t1_us,
        idle_t2_us=cfg.idle_t2_us,
        depol_p_1q=cfg.depol_p_1q,
    )
    
    lam_blind_map: Dict[str, str] = {}
    lam_value_map: Dict[str, float] = {}
    deterministic_templates: List[QuantumCircuit] = []
    
    if lambda_steps and lambda_steps > 0:
        steps = int(lambda_steps)
        lams = [float(x) for x in np.linspace(0.0, 1.0, steps)]
        lam_blind_map, lam_value_map = _make_lambda_blind_map(lams)
        
        for tok, semantic in lam_blind_map.items():
            lam = lam_value_map[tok]
            d_circs = build_bell_ramsey_deterministic(
                ramsey_delays,
                total_rounds=cfg.feedback_rounds,
                lam=lam,
                seed=cfg.seed,
                idle_t1_us=cfg.idle_t1_us,
                idle_t2_us=cfg.idle_t2_us,
                depol_p_2q=cfg.depol_p_2q,
                depol_p_1q=cfg.depol_p_1q,
            )
            for qc in d_circs:
                qc.metadata = qc.metadata or {}
                qc.metadata["blind"] = tok
                qc.metadata["kind"] = "BELL_DETERMINISTIC"
            deterministic_templates.extend(d_circs)
    
    all_circs: List[QuantumCircuit] = []
    for rep in range(cfg.n_reps):
        rep_set: List[QuantumCircuit] = []
        rep_set.extend([qc.copy() for qc in t1_circs])
        rep_set.extend([qc.copy() for qc in z_ramsey])
        rep_set.extend([qc.copy() for qc in a_ramsey])
        rep_set.extend([qc.copy() for qc in b_circs])
        rep_set.extend([qc.copy() for qc in c_circs])
        if deterministic_templates:
            rep_set.extend([qc.copy() for qc in deterministic_templates])
        
        for qc in rep_set:
            qc.metadata = qc.metadata or {}
            qc.metadata["rep"] = rep
        
        random.Random(cfg.seed + rep).shuffle(rep_set)
        all_circs.extend(rep_set)
    
    print(f"Running {len(all_circs)} circuits on Aer simulator...")
    
    pm = generate_preset_pass_manager(optimization_level=1, backend=simulator)
    transpiled = pm.run(all_circs)
    
    job = simulator.run(transpiled, shots=cfg.shots, seed_simulator=cfg.seed)
    result = job.result()
    
    decode = {v: k for k, v in CONDITION_MAP_ABCZ.items()}
    decode.update({tok: lab for tok, lab in lam_blind_map.items()})
    
    decode_path = os.path.join(out_dir, "condition_decode_map.json")
    with open(decode_path, "w", encoding="utf-8") as f:
        json.dump(decode, f, indent=2)
    
    ts = int(time.time())
    manifest = {
        "timestamp": ts,
        "backend": "aer_simulator",
        "shots": cfg.shots,
        "n_reps": cfg.n_reps,
        "seed": cfg.seed,
        "max_delay_us": cfg.max_delay_us,
        "n_delay_points": cfg.n_delay_points,
        "feedback_rounds": cfg.feedback_rounds,
        "noise_gates_per_round": cfg.noise_gates_per_round,
        "idle_t1_us": cfg.idle_t1_us,
        "idle_t2_us": cfg.idle_t2_us,
        "depol_p_1q": cfg.depol_p_1q,
        "depol_p_2q": cfg.depol_p_2q,
        "lambda_steps": int(lambda_steps or 0),
        "lambda_implementation": "DETERMINISTIC",
    }
    with open(os.path.join(out_dir, f"manifest_aer_sim_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    csv_path = os.path.join(out_dir, f"uic_rph_deterministic_aer_sim_{ts}.csv")
    fieldnames = [
        "timestamp",
        "backend",
        "blind",
        "kind",
        "rep",
        "delay_us",
        "lambda",
        "shots",
        "counts_json",
        "p00",
        "p01",
        "p10",
        "p11",
        "mutual_info_bits",
        "shannon_entropy_bits",
        "structured_dynamic_supported",
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, qc in enumerate(transpiled):
            md = qc.metadata or {}
            counts = result.get_counts(idx)
            shots = cfg.shots
            
            blind = md.get("blind", "")
            kind = md.get("kind", "")
            rep = int(md.get("rep", -1))
            delay_us = float(md.get("delay_us", np.nan))
            lam = md.get("lambda", np.nan)
            
            probs = _counts_to_probs(counts, shots)
            p00 = probs.get("00", np.nan)
            p01 = probs.get("01", np.nan)
            p10 = probs.get("10", np.nan)
            p11 = probs.get("11", np.nan)
            
            mi = np.nan
            h = np.nan
            if any(k in probs for k in ["00", "01", "10", "11"]):
                mi = _mutual_info_from_probs({k: probs.get(k, 0.0) for k in ["00", "01", "10", "11"]})
                h = _shannon_entropy({k: probs.get(k, 0.0) for k in ["00", "01", "10", "11"]})
            
            writer.writerow(
                {
                    "timestamp": ts,
                    "backend": "aer_simulator",
                    "blind": blind,
                    "kind": kind,
                    "rep": rep,
                    "delay_us": delay_us,
                    "lambda": lam,
                    "shots": shots,
                    "counts_json": json.dumps(counts),
                    "p00": p00,
                    "p01": p01,
                    "p10": p10,
                    "p11": p11,
                    "mutual_info_bits": mi,
                    "shannon_entropy_bits": h,
                    "structured_dynamic_supported": False,
                }
            )
    
    print(f"\n✅ Done.")
    print(f"CSV:     {csv_path}")
    print(f"Decode:  {decode_path}")
    print(f"Manifest:{os.path.join(out_dir, f'manifest_aer_sim_{ts}.json')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--n", type=int, default=10, help="number of reps")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=str, default="results_exp3_deterministic")
    
    ap.add_argument("--max-delay-us", type=float, default=200.0)
    ap.add_argument("--delay-points", type=int, default=15)
    
    ap.add_argument("--t1-max-delay-us", type=float, default=200.0)
    ap.add_argument("--t1-delay-points", type=int, default=10)
    
    ap.add_argument("--feedback-rounds", type=int, default=4)
    ap.add_argument("--noise-gates-per-round", type=int, default=4)
    
    ap.add_argument("--lambda-steps", type=int, default=0, help="if >0, run λ sweep with this many points in [0,1]")
    
    ap.add_argument("--idle-t1-us", type=float, default=120.0, help="idle T1 used for inserted relaxation channels")
    ap.add_argument("--idle-t2-us", type=float, default=80.0, help="idle T2 used for inserted relaxation channels")
    
    ap.add_argument("--depol-p-1q", type=float, default=0.0005)
    ap.add_argument("--depol-p-2q", type=float, default=0.002)
    
    args = ap.parse_args()
    
    cfg = ExpConfig(
        shots=args.shots,
        n_reps=args.n,
        seed=args.seed,
        max_delay_us=args.max_delay_us,
        n_delay_points=args.delay_points,
        t1_max_delay_us=args.t1_max_delay_us,
        t1_delay_points=args.t1_delay_points,
        feedback_rounds=args.feedback_rounds,
        noise_gates_per_round=args.noise_gates_per_round,
        idle_t1_us=args.idle_t1_us,
        idle_t2_us=args.idle_t2_us,
        depol_p_1q=args.depol_p_1q,
        depol_p_2q=args.depol_p_2q,
    )
    
    run_simulation(cfg, args.out, lambda_steps=args.lambda_steps)


if __name__ == "__main__":
    main()
