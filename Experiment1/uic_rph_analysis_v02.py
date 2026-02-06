"""
UIC v0.1 — RPH Analysis (BLINDED, v0.2 clean)
============================================

Per-rep upgrade:
  - Fits T2* separately for each replication (rep) per blind condition
  - Runs ANOVA + Tukey on T2* distributions (A/B/C), not on raw visibility

Inputs:
  - CSV output from uic_rph_hardware_v02.py

Workflow:
  1) Run blinded:
       python uic_rph_analysis_v02.py --csv uic_run_01/uic_rph_blinded_*.csv --out uic_run_01/analysis_blinded
  2) After stats, decode (optional):
       python uic_rph_analysis_v02.py --csv ... --out ... --decode uic_run_01/condition_decode_map.json

Requirements:
  - numpy, pandas, scipy, matplotlib, statsmodels
"""

import argparse
import json
import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def exp_decay(t, a, T):
    return a * np.exp(-t / T)


def ramsey_model(t, a, T2s, w, phi, c):
    return a * np.exp(-t / T2s) * np.cos(w * t + phi) + c


def cohen_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["delay_us"] = pd.to_numeric(df["delay_us"], errors="coerce")
    df["shots"] = pd.to_numeric(df["shots"], errors="coerce")
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce")
    return df


def row_visibility(counts: Dict[str, int], shots: int) -> float:
    keys = list(counts.keys())
    if any(k in counts for k in ["0", "1"]) and not any(len(k) == 2 for k in keys):
        c0 = counts.get("0", 0)
        c1 = counts.get("1", 0)
        return (c0 - c1) / shots
    c00 = counts.get("00", 0)
    c01 = counts.get("01", 0)
    c10 = counts.get("10", 0)
    c11 = counts.get("11", 0)
    return ((c00 + c11) - (c01 + c10)) / shots


def fit_t1(df: pd.DataFrame) -> Tuple[float, Dict]:
    d = df[(df["blind"] == "Z") & (df["kind"] == "T1")].copy()
    if d.empty:
        return np.nan, {"error": "No T1 data found"}

    p1 = []
    t = []
    for _, row in d.iterrows():
        counts = json.loads(row["counts_json"])
        shots = int(row["shots"])
        p1.append(counts.get("1", 0) / shots)
        t.append(float(row["delay_us"]))
    t = np.array(t, dtype=float)
    p1 = np.array(p1, dtype=float)

    p0 = [max(p1[0], 1e-3), max(t.max() / 2, 1.0)]
    try:
        popt, pcov = curve_fit(exp_decay, t, p1, p0=p0, maxfev=10000)
        a, T1 = popt
        return float(T1), {"a": float(a), "cov": pcov.tolist(), "n": int(len(t))}
    except Exception as e:
        return np.nan, {"error": str(e), "n": int(len(t))}


def fit_t2_star_for_rep(rep_df: pd.DataFrame) -> Tuple[float, Dict]:
    t = rep_df["delay_us"].to_numpy(dtype=float)
    v = rep_df["visibility"].to_numpy(dtype=float)

    idx = np.argsort(t)
    t = t[idx]; v = v[idx]

    a0 = float(np.clip(np.max(np.abs(v)), 0.05, 1.0))
    T0 = float(max(t.max() / 2, 5.0))
    w0 = 2 * math.pi * (1.0 / max(t.max(), 1.0))
    phi0 = 0.0
    c0 = float(np.mean(v[-max(3, len(v)//10):])) if len(v) >= 3 else 0.0
    p0 = [a0, T0, w0, phi0, c0]

    try:
        popt, pcov = curve_fit(ramsey_model, t, v, p0=p0, maxfev=20000)
        a, T2s, w, phi, c = popt
        return float(abs(T2s)), {"a": float(a), "w": float(w), "phi": float(phi), "c": float(c)}
    except Exception as e:
        return np.nan, {"error": str(e)}


def plot_rep_fit(rep_df: pd.DataFrame, T2s: float, out_png: str):
    t = rep_df["delay_us"].to_numpy(dtype=float)
    v = rep_df["visibility"].to_numpy(dtype=float)
    idx = np.argsort(t)
    t = t[idx]; v = v[idx]

    a0 = float(np.clip(np.max(np.abs(v)), 0.05, 1.0))
    T0 = float(max(t.max() / 2, 5.0))
    w0 = 2 * math.pi * (1.0 / max(t.max(), 1.0))
    p0 = [a0, T0, w0, 0.0, float(np.mean(v))]
    try:
        popt, _ = curve_fit(ramsey_model, t, v, p0=p0, maxfev=20000)
    except Exception:
        return

    tt = np.linspace(t.min(), t.max(), 400)
    yy = ramsey_model(tt, *popt)

    plt.figure()
    plt.scatter(t, v)
    plt.plot(tt, yy)
    plt.xlabel("delay (µs)")
    plt.ylabel("visibility")
    plt.title(f"T2* ≈ {T2s:.2f} µs")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--decode", default=None, help="path to condition_decode_map.json (only after stats)")
    ap.add_argument("--out", default="uic_rph_analysis_out")
    ap.add_argument("--plot-reps", action="store_true", help="save one per-rep fit plot per blind")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_df(args.csv)

    # Fit T1 (Z)
    T1, t1_info = fit_t1(df)
    with open(os.path.join(args.out, "t1_fit.json"), "w", encoding="utf-8") as f:
        json.dump({"T1_us": T1, "details": t1_info}, f, indent=2)

    # Visibility dataframe (exclude T1; keep Ramsey/Bell)
    vis_rows = []
    for _, row in df.iterrows():
        bl = row.get("blind")
        if bl not in ["A", "B", "C", "Z"]:
            continue
        kind = str(row.get("kind", ""))
        if "T1" in kind:
            continue
        if not ("RAMSEY" in kind or "BELL" in kind):
            continue
        rep = row.get("rep", np.nan)
        if pd.isna(rep):
            continue
        rep = int(rep)
        if rep < 0:
            continue

        counts = json.loads(row["counts_json"])
        shots = int(row["shots"])
        v = row_visibility(counts, shots)

        vis_rows.append({
            "blind": bl,
            "rep": rep,
            "delay_us": float(row["delay_us"]),
            "visibility": float(v),
        })

    vis_df = pd.DataFrame(vis_rows)
    if vis_df.empty:
        raise RuntimeError("No visibility data found. Check runner output and CSV content.")

    # Fit T2* per (blind, rep)
    t2_rows = []
    for (bl, rep), g in vis_df.groupby(["blind", "rep"]):
        T2s, info = fit_t2_star_for_rep(g)
        t2_rows.append({"blind": bl, "rep": rep, "T2*_us": T2s, "fit_error": info.get("error", "")})

    t2_rep = pd.DataFrame(t2_rows)
    t2_rep = t2_rep[np.isfinite(t2_rep["T2*_us"])].copy()
    t2_rep.to_csv(os.path.join(args.out, "t2star_per_rep_blinded.csv"), index=False)

    # Optional quick plots: one rep per blind
    if args.plot_reps:
        for bl in ["A", "B", "C", "Z"]:
            sub = t2_rep[t2_rep["blind"] == bl].sort_values("rep")
            if sub.empty:
                continue
            rep0 = int(sub.iloc[0]["rep"])
            T2s = float(sub.iloc[0]["T2*_us"])
            g = vis_df[(vis_df["blind"] == bl) & (vis_df["rep"] == rep0)].copy()
            plot_rep_fit(g, T2s, os.path.join(args.out, f"repfit_{bl}_rep{rep0}.png"))

    # Primary stats: ANOVA on per-rep T2* across A/B/C
    t2_abc = t2_rep[t2_rep["blind"].isin(["A", "B", "C"])].copy()
    model = ols("Q('T2*_us') ~ C(blind)", data=t2_abc).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(os.path.join(args.out, "anova_t2star.csv"))

    # Tukey HSD on T2*
    tuk = pairwise_tukeyhsd(endog=t2_abc["T2*_us"], groups=t2_abc["blind"], alpha=0.05)
    with open(os.path.join(args.out, "tukey_t2star.txt"), "w", encoding="utf-8") as f:
        f.write(str(tuk))

    # Effect sizes on T2*
    d_ab = cohen_d(t2_abc[t2_abc["blind"] == "A"]["T2*_us"], t2_abc[t2_abc["blind"] == "B"]["T2*_us"])
    d_ac = cohen_d(t2_abc[t2_abc["blind"] == "A"]["T2*_us"], t2_abc[t2_abc["blind"] == "C"]["T2*_us"])
    d_bc = cohen_d(t2_abc[t2_abc["blind"] == "B"]["T2*_us"], t2_abc[t2_abc["blind"] == "C"]["T2*_us"])
    with open(os.path.join(args.out, "effect_sizes_t2star.json"), "w", encoding="utf-8") as f:
        json.dump({"d_A_vs_B": d_ab, "d_A_vs_C": d_ac, "d_B_vs_C": d_bc}, f, indent=2)

    # Decode (optional) AFTER stats
    if args.decode:
        with open(args.decode, "r", encoding="utf-8") as f:
            decode_map = json.load(f)  # e.g., {'A':'isolated', ...}
        t2_dec = t2_rep.copy()
        t2_dec["decoded"] = t2_dec["blind"].map(decode_map)
        t2_dec.to_csv(os.path.join(args.out, "t2star_per_rep_decoded.csv"), index=False)

    print(f"Saved: {args.out}/t2star_per_rep_blinded.csv")
    print(f"Saved: {args.out}/anova_t2star.csv")
    print(f"Saved: {args.out}/tukey_t2star.txt")
    print(f"Saved: {args.out}/effect_sizes_t2star.json")
    if args.decode:
        print(f"Saved: {args.out}/t2star_per_rep_decoded.csv (post-stats decode)")


if __name__ == "__main__":
    main()
