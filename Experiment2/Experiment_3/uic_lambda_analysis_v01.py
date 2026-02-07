"""
UIC Lambda-Sweep Analysis v0.1: Post-Processing Mixture
========================================================

Takes pure quantum Bell measurements and mixes them with classical
reference probabilities to simulate hidden variable models.

For each lambda:
  P_mixed = (1-λ)*P_quantum + λ*P_classical

Then fits T2* decay curves to see if T2*(λ) behaves linearly.
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exp_decay_offset(t, a, T, c):
    """Exponential decay: V(t) = a * exp(-t/T) + c"""
    return a * np.exp(-t / T) + c


def compute_visibility(p00: float, p11: float, p01: float, p10: float) -> float:
    """Parity/visibility: (P00 + P11) - (P01 + P10)"""
    return (p00 + p11) - (p01 + p10)


def mix_probabilities(
        p_quantum: Dict[str, float],
        p_classical: Dict[str, float],
        lam: float
) -> Dict[str, float]:
    """
    Mix quantum and classical probabilities:
    P_mixed = (1-λ)*P_quantum + λ*P_classical
    """
    mixed = {}
    for outcome in ["p00", "p11", "p01", "p10"]:
        pq = p_quantum.get(outcome, 0.0)
        pc = p_classical.get(outcome, 0.0)
        mixed[outcome] = (1.0 - lam) * pq + lam * pc
    return mixed


def fit_t2_star(delays: np.ndarray, visibility: np.ndarray) -> Tuple[float, Dict]:
    """
    Fit V(t) = a*exp(-t/T) + c
    Returns (T2*, metadata)
    """
    # Remove NaNs
    mask = np.isfinite(delays) & np.isfinite(visibility)
    t = delays[mask]
    y = visibility[mask]

    if len(t) < 6:
        return (np.nan, {"reason": "too_few_points"})

    if np.std(y) < 1e-6:
        return (np.nan, {"reason": "constant_visibility"})

    # Initial guesses
    c0 = float(np.median(y[-max(2, len(y) // 4):]))
    a0 = float(y[0] - c0)
    T0 = max(1.0, float(np.max(t)) / 3.0)

    # Bounds
    bounds = ([-2.0, 1e-6, -1.5], [2.0, 1e6, 1.5])

    try:
        popt, _ = curve_fit(
            exp_decay_offset, t, y,
            p0=[a0, T0, c0],
            bounds=bounds,
            maxfev=20000
        )
        a, T, c = [float(x) for x in popt]

        # R^2
        yhat = exp_decay_offset(t, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return (T, {"a": a, "c": c, "r2": r2})
    except Exception as e:
        return (np.nan, {"reason": f"fit_failed: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to lambda sweep CSV")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)

    # Filter lambda sweep circuits
    lambda_df = df[df["kind"] == "BELL_LAMBDA_SWEEP"].copy()

    if lambda_df.empty:
        raise SystemExit("No BELL_LAMBDA_SWEEP circuits found!")

    # Parse classical reference probabilities
    def parse_classical(ref_json: str) -> Dict[str, float]:
        try:
            raw = json.loads(ref_json)
            return {
                "p00": raw.get("00", 0.0),
                "p11": raw.get("11", 0.0),
                "p01": raw.get("01", 0.0),
                "p10": raw.get("10", 0.0),
            }
        except:
            return {"p00": 0.5, "p11": 0.5, "p01": 0.0, "p10": 0.0}

    lambda_df["classical_probs"] = lambda_df["classical_ref"].apply(parse_classical)

    # Compute mixed probabilities
    def mix_row(row):
        p_quantum = {
            "p00": row["p00"],
            "p11": row["p11"],
            "p01": row["p01"],
            "p10": row["p10"],
        }
        p_classical = row["classical_probs"]
        lam = row["lambda"]
        return mix_probabilities(p_quantum, p_classical, lam)

    mixed = lambda_df.apply(mix_row, axis=1, result_type="expand")
    lambda_df["p00_mixed"] = mixed["p00"]
    lambda_df["p11_mixed"] = mixed["p11"]
    lambda_df["p01_mixed"] = mixed["p01"]
    lambda_df["p10_mixed"] = mixed["p10"]

    # Compute visibility
    lambda_df["vis_mixed"] = lambda_df.apply(
        lambda r: compute_visibility(
            r["p00_mixed"], r["p11_mixed"], r["p01_mixed"], r["p10_mixed"]
        ),
        axis=1
    )

    # Fit T2* per lambda per rep
    results = []
    for (lam, rep), group in lambda_df.groupby(["lambda", "rep"]):
        group = group.sort_values("delay_us")
        delays = group["delay_us"].to_numpy()
        vis = group["vis_mixed"].to_numpy()

        T2, meta = fit_t2_star(delays, vis)

        results.append({
            "lambda": float(lam),
            "rep": int(rep),
            "t2star_us": T2,
            "n_points": len(group),
            "fit_meta": json.dumps(meta),
        })

    # Save per-rep results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out, "t2star_lambda_per_rep.csv"), index=False)

    # Summarize per lambda
    summary = (
        results_df[np.isfinite(results_df["t2star_us"])]
        .groupby("lambda")["t2star_us"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out, "t2star_lambda_summary.csv"), index=False)

    # Simple linear regression
    x = summary["lambda"].to_numpy()
    y = summary["mean"].to_numpy()

    if len(x) >= 3:
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = m * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        with open(os.path.join(args.out, "lambda_regression.txt"), "w", encoding="utf-8") as f:
            f.write("Linear regression: T2*(lambda) = m*lambda + b\n")
            f.write(f"slope (m): {m:.6g} us per lambda\n")
            f.write(f"intercept (b): {b:.6g} us\n")
            f.write(f"R^2: {r2:.6g}\n")
            f.write("\n")
            if r2 < 0.9:
                f.write("⚠️ Low R^2 suggests NON-LINEAR behavior (potential quantum violation!)\n")
            else:
                f.write("✅ High R^2 suggests linear interpolation (classical mixing)\n")

    print(f"\n✅ Analysis complete!")
    print(f"Per-rep:  {os.path.join(args.out, 't2star_lambda_per_rep.csv')}")
    print(f"Summary:  {os.path.join(args.out, 't2star_lambda_summary.csv')}")
    print(f"Regression: {os.path.join(args.out, 'lambda_regression.txt')}")
    print("\nLambda Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
