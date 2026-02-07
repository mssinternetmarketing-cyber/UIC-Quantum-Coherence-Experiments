"""
Lambda Sweep Analysis: T2* vs Classical Mixing Parameter
=========================================================

Analyzes how T2* degrades with increasing lambda (classical noise mixing).
Tests the hypothesis: Higher lambda → more classical → faster decoherence.
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
    Returns (T2*, amplitude, metadata)
    """
    mask = np.isfinite(delays) & np.isfinite(visibility)
    t = delays[mask]
    y = visibility[mask]

    if len(t) < 6:
        return (np.nan, np.nan, {"reason": "too_few_points"})

    if np.std(y) < 1e-6:
        return (np.nan, np.nan, {"reason": "constant_visibility"})

    c0 = float(np.median(y[-max(2, len(y) // 4):]))
    a0 = float(y[0] - c0)
    T0 = max(1.0, float(np.max(t)) / 3.0)

    bounds = ([-2.0, 1e-6, -1.5], [2.0, 1e6, 1.5])

    try:
        popt, _ = curve_fit(
            exp_decay_offset, t, y,
            p0=[a0, T0, c0],
            bounds=bounds,
            maxfev=20000
        )
        a, T, c = [float(x) for x in popt]

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
    ap.add_argument("--rounds", type=int, default=5, help="which round count to analyze (default: 5)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load data
    df = pd.read_csv(args.csv)

    # Filter lambda sweep circuits at specific round count
    lambda_df = df[
        (df["kind"] == "BELL_LAMBDA_SWEEP") &
        (df["rounds"] == args.rounds)
        ].copy()

    if lambda_df.empty:
        raise SystemExit(f"No BELL_LAMBDA_SWEEP circuits found at rounds={args.rounds}!")

    print(f"Analyzing {len(lambda_df)} circuits at {args.rounds} rounds...")

    # Compute visibility
    lambda_df["vis"] = lambda_df.apply(
        lambda r: compute_visibility(r["p00"], r["p11"], r["p01"], r["p10"]),
        axis=1
    )

    # Fit T2* per lambda per rep
    results = []
    for (lam, rep), group in lambda_df.groupby(["lambda", "rep"]):
        group = group.sort_values("delay_us")
        delays = group["delay_us"].to_numpy()
        vis = group["vis"].to_numpy()

        T2, amplitude, meta = fit_t2_star(delays, vis)

        # Quality filter
        fit_meta = json.loads(meta) if isinstance(meta, str) else meta
        r2 = fit_meta.get("r2", np.nan)

        if np.isfinite(T2) and T2 < 500 and r2 > 0.3:
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
    results_df.to_csv(os.path.join(args.out, f"t2star_lambda_per_rep_rounds{args.rounds}.csv"), index=False)

    # Summarize per lambda
    valid = results_df[np.isfinite(results_df["t2star_us"])].copy()

    if valid.empty:
        raise SystemExit("No valid T2* fits! Try relaxing quality filters.")

    summary = (
        valid.groupby("lambda")["t2star_us"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out, f"t2star_lambda_summary_rounds{args.rounds}.csv"), index=False)

    # Plot: T2* vs Lambda
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: T2* vs Lambda (linear scale)
    ax1.errorbar(summary["lambda"], summary["mean"], yerr=summary["std"],
                 fmt='o-', capsize=5, markersize=8, linewidth=2, color='purple')
    ax1.set_xlabel("λ (Classical Mixing Parameter)", fontsize=12)
    ax1.set_ylabel("T2* (μs)", fontsize=12)
    ax1.set_title(f"T2* Degradation vs Classical Mixing (Rounds={args.rounds})", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Annotate quantum/classical endpoints
    lambda_0 = summary[summary["lambda"] == 0.0]["mean"].values[0] if 0.0 in summary["lambda"].values else np.nan
    lambda_1 = summary[summary["lambda"] == 1.0]["mean"].values[0] if 1.0 in summary["lambda"].values else np.nan

    if np.isfinite(lambda_0):
        ax1.axhline(lambda_0, color='blue', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Pure quantum (λ=0): {lambda_0:.1f} μs')
    if np.isfinite(lambda_1):
        ax1.axhline(lambda_1, color='red', linestyle='--', linewidth=1, alpha=0.5,
                    label=f'Pure classical (λ=1): {lambda_1:.1f} μs')
    ax1.legend(fontsize=9)

    # Right: 1/T2* vs Lambda (decoherence rate)
    summary["decoherence_rate"] = 1 / summary["mean"]
    summary["decoherence_rate_err"] = summary["std"] / (summary["mean"] ** 2)

    ax2.errorbar(summary["lambda"], summary["decoherence_rate"], yerr=summary["decoherence_rate_err"],
                 fmt='o-', capsize=5, markersize=8, linewidth=2, color='green')
    ax2.set_xlabel("λ (Classical Mixing Parameter)", fontsize=12)
    ax2.set_ylabel("1/T2* (μs⁻¹)", fontsize=12)
    ax2.set_title(f"Decoherence Rate vs Classical Mixing (Rounds={args.rounds})", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Linear fit to test if decoherence rate ∝ lambda
    x = summary["lambda"].to_numpy()
    y = summary["decoherence_rate"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    ax2.plot(x, m * x + b, 'r--', linewidth=2, label=f'Linear fit: slope={m:.4f} μs⁻¹/λ')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"lambda_analysis_plot_rounds{args.rounds}.png"), dpi=150)
    print(f"Plot saved: {os.path.join(args.out, f'lambda_analysis_plot_rounds{args.rounds}.png')}")

    # Compute visibility at t=0 and t=max per lambda (for sanity check)
    vis_stats = []
    for lam, group in lambda_df.groupby("lambda"):
        vis_t0 = group[group["delay_us"] == 0.0]["vis"].mean()
        vis_tmax = group[group["delay_us"] == group["delay_us"].max()]["vis"].mean()
        vis_stats.append({
            "lambda": float(lam),
            "vis_t0": vis_t0,
            "vis_tmax": vis_tmax,
            "delta_vis": vis_t0 - vis_tmax
        })

    vis_stats_df = pd.DataFrame(vis_stats)
    vis_stats_df.to_csv(os.path.join(args.out, f"visibility_stats_rounds{args.rounds}.csv"), index=False)

    # Write summary report
    with open(os.path.join(args.out, f"lambda_report_rounds{args.rounds}.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"LAMBDA SWEEP ANALYSIS (Rounds={args.rounds})\n")
        f.write("=" * 70 + "\n\n")

        f.write("Goal: Demonstrate how classical mixing (λ) affects decoherence.\n")
        f.write("      λ=0 → Pure quantum entanglement\n")
        f.write("      λ=1 → Pure classical correlation\n\n")

        f.write("Results:\n")
        f.write("-" * 70 + "\n")
        for _, row in summary.iterrows():
            f.write(f"λ = {row['lambda']:.2f}  |  ")
            f.write(f"T2*: {row['mean']:6.2f} ± {row['std']:5.2f} μs  |  ")
            f.write(f"1/T2*: {1 / row['mean']:.4f} μs⁻¹\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("=" * 70 + "\n\n")

        # Check if T2* decreases with lambda
        t2_values = summary.sort_values("lambda")["mean"].to_numpy()
        is_decreasing = all(t2_values[i] >= t2_values[i + 1] for i in range(len(t2_values) - 1))

        if is_decreasing:
            f.write("✅ T2* decreases monotonically with λ\n")
            f.write("   → Classical mixing accelerates decoherence\n\n")
        else:
            f.write("⚠️ T2* does not decrease monotonically with λ\n")
            f.write("   → Unexpected behavior; check data quality\n\n")

        # Compare quantum vs classical
        if np.isfinite(lambda_0) and np.isfinite(lambda_1):
            ratio = lambda_0 / lambda_1
            f.write(f"Pure quantum T2* (λ=0): {lambda_0:.2f} μs\n")
            f.write(f"Pure classical T2* (λ=1): {lambda_1:.2f} μs\n")
            f.write(f"Ratio: {ratio:.2f}x\n\n")

            if ratio > 1.5:
                f.write("✅ Quantum states persist significantly longer than classical\n")
                f.write("   → Validates thermodynamic decoherence model\n\n")
            else:
                f.write("⚠️ Quantum/classical difference smaller than expected\n")
                f.write("   → May indicate high gate noise floor\n\n")

        f.write(f"Linear model: 1/T2* = {m:.4f} * λ + {b:.4f}\n")
        f.write(f"Slope: {m:.4f} μs⁻¹ per unit λ\n")
        f.write(f"This represents the decoherence cost of classical mixing.\n\n")

        f.write("Visibility decay:\n")
        f.write("-" * 70 + "\n")
        for _, row in vis_stats_df.iterrows():
            f.write(f"λ = {row['lambda']:.2f}  |  ")
            f.write(f"Vis(t=0): {row['vis_t0']:.3f}  |  ")
            f.write(f"Vis(t=max): {row['vis_tmax']:.3f}  |  ")
            f.write(f"Δ: {row['delta_vis']:.3f}\n")

    print(f"\n✅ Lambda analysis complete!")
    print(f"Per-rep:  {os.path.join(args.out, f't2star_lambda_per_rep_rounds{args.rounds}.csv')}")
    print(f"Summary:  {os.path.join(args.out, f't2star_lambda_summary_rounds{args.rounds}.csv')}")
    print(f"Report:   {os.path.join(args.out, f'lambda_report_rounds{args.rounds}.txt')}")
    print("\nLambda Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
