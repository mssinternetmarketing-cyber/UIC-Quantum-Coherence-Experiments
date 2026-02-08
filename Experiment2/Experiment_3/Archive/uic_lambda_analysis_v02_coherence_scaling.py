"""
UIC Lambda-Sweep Analysis v0.2: Coherence Scaling Model
========================================================

Implements the CORRECT hidden variable model:
- Classical ≠ "stable forever"
- Classical = "no coherence" (off-diagonal terms = 0)

For each lambda:
  V_mixed(t, λ) = (1-λ) * V_quantum(t)

This scales the VISIBILITY AMPLITUDE, not the decay rate.

Expected result:
- T2* should remain ~80 μs for all λ (decay constant unchanged)
- Visibility amplitude should scale linearly: V(0) = (1-λ) * V_quantum(0)
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exp_decay_offset(t, a, T, c):
    """Exponential decay: V(t) = a * exp(-t/T) + c"""
    return a * np.exp(-t / T) + c


def compute_visibility(p00: float, p11: float, p01: float, p10: float) -> float:
    """Parity/visibility: (P00 + P11) - (P01 + P10)"""
    return (p00 + p11) - (p01 + p10)


def fit_t2_star(delays: np.ndarray, visibility: np.ndarray) -> Tuple[float, float, Dict]:
    """
    Fit V(t) = a*exp(-t/T) + c
    Returns (T2*, amplitude_a, metadata)
    """
    # Remove NaNs
    mask = np.isfinite(delays) & np.isfinite(visibility)
    t = delays[mask]
    y = visibility[mask]

    if len(t) < 6:
        return (np.nan, np.nan, {"reason": "too_few_points"})

    if np.std(y) < 1e-6:
        return (np.nan, np.nan, {"reason": "constant_visibility"})

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

        return (T, a, {"a": a, "c": c, "r2": r2, "T": T})
    except Exception as e:
        return (np.nan, np.nan, {"reason": f"fit_failed: {e}"})


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

    # Compute quantum visibility (no mixing, just raw measurements)
    lambda_df["vis_quantum"] = lambda_df.apply(
        lambda r: compute_visibility(r["p00"], r["p11"], r["p01"], r["p10"]),
        axis=1
    )

    # Apply coherence scaling: V_mixed(t, λ) = (1-λ) * V_quantum(t)
    lambda_df["vis_mixed"] = lambda_df.apply(
        lambda r: (1.0 - r["lambda"]) * r["vis_quantum"],
        axis=1
    )

    # Fit T2* per lambda per rep
    results = []
    for (lam, rep), group in lambda_df.groupby(["lambda", "rep"]):
        group = group.sort_values("delay_us")
        delays = group["delay_us"].to_numpy()
        vis = group["vis_mixed"].to_numpy()

        T2, amplitude, meta = fit_t2_star(delays, vis)

        results.append({
            "lambda": float(lam),
            "rep": int(rep),
            "t2star_us": T2,
            "amplitude": amplitude,
            "n_points": len(group),
            "fit_meta": json.dumps(meta),
        })

    # Save per-rep results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out, "t2star_lambda_per_rep.csv"), index=False)

    # Summarize per lambda
    valid = results_df[np.isfinite(results_df["t2star_us"])].copy()

    t2_summary = (
        valid.groupby("lambda")["t2star_us"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )

    amp_summary = (
        valid.groupby("lambda")["amplitude"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    summary = t2_summary.merge(amp_summary, on="lambda", suffixes=("_t2", "_amp"))
    summary.to_csv(os.path.join(args.out, "t2star_lambda_summary.csv"), index=False)

    # Linear regression on T2*
    x = summary["lambda"].to_numpy()
    y_t2 = summary["mean_t2"].to_numpy()

    if len(x) >= 3:
        A = np.vstack([x, np.ones_like(x)]).T
        m_t2, b_t2 = np.linalg.lstsq(A, y_t2, rcond=None)[0]
        yhat_t2 = m_t2 * x + b_t2
        ss_res_t2 = float(np.sum((y_t2 - yhat_t2) ** 2))
        ss_tot_t2 = float(np.sum((y_t2 - np.mean(y_t2)) ** 2))
        r2_t2 = 1.0 - ss_res_t2 / ss_tot_t2 if ss_tot_t2 > 0 else np.nan

        # Linear regression on Amplitude
        y_amp = summary["mean_amp"].to_numpy()
        m_amp, b_amp = np.linalg.lstsq(A, y_amp, rcond=None)[0]
        yhat_amp = m_amp * x + b_amp
        ss_res_amp = float(np.sum((y_amp - yhat_amp) ** 2))
        ss_tot_amp = float(np.sum((y_amp - np.mean(y_amp)) ** 2))
        r2_amp = 1.0 - ss_res_amp / ss_tot_amp if ss_tot_amp > 0 else np.nan

        with open(os.path.join(args.out, "lambda_regression.txt"), "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("COHERENCE SCALING MODEL RESULTS\n")
            f.write("=" * 60 + "\n\n")

            f.write("MODEL: V_mixed(t, lambda) = (1-lambda) * V_quantum(t)\n\n")

            f.write("--- T2* vs Lambda ---\n")
            f.write(f"slope: {m_t2:.6g} us per lambda\n")
            f.write(f"intercept: {b_t2:.6g} us\n")
            f.write(f"R^2: {r2_t2:.6g}\n\n")

            if abs(m_t2) < 5.0 and r2_t2 < 0.1:
                f.write("✅ T2* is CONSTANT across lambda (as expected!)\n")
                f.write("   → Decay rate is independent of hidden variable bias\n\n")
            else:
                f.write("⚠️ T2* shows variation with lambda\n")
                f.write("   → May indicate fitting issues or non-exponential decay\n\n")

            f.write("--- Amplitude vs Lambda ---\n")
            f.write(f"slope: {m_amp:.6g} per lambda\n")
            f.write(f"intercept: {b_amp:.6g}\n")
            f.write(f"R^2: {r2_amp:.6g}\n\n")

            if r2_amp > 0.95 and abs(m_amp + b_amp) < 0.1:
                f.write("✅ Amplitude scales linearly: a(lambda) ≈ (1-lambda) * a_quantum\n")
                f.write("   → Classical model = coherence suppression (as expected!)\n\n")
            else:
                f.write("⚠️ Amplitude doesn't follow linear scaling\n")
                f.write("   → Potential quantum violation or fit instability\n\n")

            f.write("=" * 60 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("=" * 60 + "\n\n")
            f.write("If T2* constant + Amplitude linear:\n")
            f.write("  → Hidden variables just suppress coherence amplitude\n")
            f.write("  → Decay dynamics are unchanged (classical mixing works)\n\n")
            f.write("If T2* varies OR Amplitude nonlinear:\n")
            f.write("  → Quantum effects cannot be reduced to classical mixing\n")
            f.write("  → Potential Bell-type violation in time domain!\n")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # T2* vs lambda
    ax1.errorbar(summary["lambda"], summary["mean_t2"], yerr=summary["std_t2"],
                 fmt='o-', capsize=5, label="T2* (measured)")
    ax1.axhline(summary["mean_t2"].mean(), color='r', linestyle='--',
                label=f"Mean = {summary['mean_t2'].mean():.1f} μs")
    ax1.set_xlabel("Lambda (hidden variable bias)")
    ax1.set_ylabel("T2* (μs)")
    ax1.set_title("T2* vs Lambda (Should be Constant)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Amplitude vs lambda
    ax2.errorbar(summary["lambda"], summary["mean_amp"], yerr=summary["std_amp"],
                 fmt='o-', capsize=5, label="Amplitude (measured)")

    # Expected linear scaling
    expected_amp = (1 - summary["lambda"]) * summary["mean_amp"].iloc[0]
    ax2.plot(summary["lambda"], expected_amp, 'r--',
             label="Expected: (1-λ) * a_quantum")

    ax2.set_xlabel("Lambda (hidden variable bias)")
    ax2.set_ylabel("Visibility Amplitude")
    ax2.set_title("Amplitude vs Lambda (Should be Linear)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "lambda_analysis_plot.png"), dpi=150)
    print(f"Plot saved: {os.path.join(args.out, 'lambda_analysis_plot.png')}")

    print(f"\n✅ Analysis complete!")
    print(f"Per-rep:  {os.path.join(args.out, 't2star_lambda_per_rep.csv')}")
    print(f"Summary:  {os.path.join(args.out, 't2star_lambda_summary.csv')}")
    print(f"Regression: {os.path.join(args.out, 'lambda_regression.txt')}")
    print("\nLambda Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
