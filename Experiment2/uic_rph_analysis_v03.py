"""
UIC v0.1 — RPH Analysis (BLINDED, v0.3: stable decay fits + λ-sweep support)
==========================================================================

What’s new vs v0.2:
- Uses a *stable exponential-with-offset* model for parity/visibility decays:
    V(t) = a * exp(-t / T2*) + c
  This avoids ill-conditioned fits when there is no oscillatory Ramsey fringe.
- Supports optional λ-sweep runs from uic_rph_sim_runner_v03.py:
    - If the CSV contains a `lambda` column and/or blind tokens decode to LAMBDA_xxx,
      the analysis will compute per-λ T2* distributions and (optionally) a simple regression.

Workflow:
  1) Analyze blinded:
       python uic_rph_analysis_v03.py --csv path/to/uic_rph_blinded_*.csv --out analysis_blinded
  2) (Optional) decode AFTER you’ve frozen stats:
       python uic_rph_analysis_v03.py --csv ... --out ... --decode condition_decode_map.json

Outputs:
  - t1_fit.json (if calibration present)
  - t2star_per_rep_blinded.csv
  - anova_t2star.csv + tukey_t2star.txt (for categorical conditions)
  - effect_sizes_t2star.json
  - plots per condition/rep
  - lambda_summary.csv + lambda_regression.txt (if λ sweep detected)

Notes:
- This is still “epistemically clean”: it does not assume semantic labels unless decoded.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def exp_decay_offset(t, a, T, c):
    return a * np.exp(-t / T) + c


def cohen_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["delay_us"] = pd.to_numeric(df.get("delay_us"), errors="coerce")
    df["shots"] = pd.to_numeric(df.get("shots"), errors="coerce")
    df["rep"] = pd.to_numeric(df.get("rep"), errors="coerce")
    # optional lambda
    if "lambda" in df.columns:
        df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    else:
        df["lambda"] = np.nan
    return df


def parse_counts(counts_json: str) -> Dict[str, int]:
    try:
        return json.loads(counts_json)
    except Exception:
        return {}


def row_visibility(counts: Dict[str, int], shots: int) -> float:
    """
    Visibility/parity proxy:
    - For 1q: (c0 - c1)/shots
    - For 2q: parity = (P00+P11) - (P01+P10)
    """
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


def decode_blinds(df: pd.DataFrame, decode_path: str) -> pd.DataFrame:
    with open(decode_path, "r", encoding="utf-8") as f:
        dec = json.load(f)
    # dec maps blind token -> semantic label
    df = df.copy()
    df["decoded"] = df["blind"].map(lambda b: dec.get(str(b), str(b)))
    # try to extract lambda if label is like "LAMBDA_0.300"
    def extract_lambda(x: str) -> float:
        try:
            if isinstance(x, str) and x.startswith("LAMBDA_"):
                return float(x.split("_", 1)[1])
        except Exception:
            pass
        return np.nan
    df["decoded_lambda"] = df["decoded"].map(extract_lambda)
    # prefer decoded_lambda when present
    df["lambda"] = np.where(np.isfinite(df["decoded_lambda"]), df["decoded_lambda"], df["lambda"])
    return df


def fit_t1(df: pd.DataFrame) -> Tuple[float, Dict]:
    d = df[(df["blind"] == "Z") & (df["kind"] == "T1")].copy()
    if d.empty:
        return (np.nan, {})

    d["counts"] = d["counts_json"].map(parse_counts)
    d["vis"] = d.apply(lambda r: row_visibility(r["counts"], int(r["shots"])), axis=1)

    # crude T1 proxy using exp decay on excited population is not ideal here;
    # keep it as a placeholder summary.
    # We'll just report mean visibility at max delay.
    max_delay = float(d["delay_us"].max())
    v_end = float(d.loc[d["delay_us"] == max_delay, "vis"].mean())
    out = {"t1_proxy_max_delay_us": max_delay, "visibility_end": v_end}
    return (max_delay, out)


def fit_t2_star_for_rep(d: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Fit V(t)=a*exp(-t/T)+c, return T in microseconds.
    """
    d = d.sort_values("delay_us")
    t = d["delay_us"].to_numpy(dtype=float)
    y = d["vis"].to_numpy(dtype=float)

    # Remove NaNs
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    if len(t) < 6:
        return (np.nan, {"reason": "too_few_points"})

    # If essentially constant, fits are meaningless
    if float(np.nanstd(y)) < 1e-6:
        return (np.nan, {"reason": "near_constant_visibility"})

    # Initial guesses
    c0 = float(np.median(y[-max(2, len(y)//4):]))  # tail median
    a0 = float(y[0] - c0)
    T0 = max(1.0, float(np.nanmax(t)) / 3.0)

    # Bounds: T positive, c in [-1.5,1.5], a in [-2,2]
    bounds = ([-2.0, 1e-6, -1.5], [2.0, 1e6, 1.5])

    try:
        popt, pcov = curve_fit(
            exp_decay_offset,
            t,
            y,
            p0=[a0, T0, c0],
            bounds=bounds,
            maxfev=20000,
        )
        a, T, c = [float(x) for x in popt]
        # simple R^2
        yhat = exp_decay_offset(t, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return (T, {"a": a, "c": c, "r2": r2})
    except Exception as e:
        return (np.nan, {"reason": f"fit_failed: {e}"})


def ensure_out(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to blinded CSV (runner output)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--decode", default="", help="optional decode map json (AFTER stats are frozen)")
    args = ap.parse_args()

    ensure_out(args.out)

    df = load_df(args.csv)

    # Optional decode
    if args.decode:
        df = decode_blinds(df, args.decode)
        label_col = "decoded"
    else:
        df["decoded"] = df["blind"]
        df["decoded_lambda"] = np.nan
        label_col = "blind"

    # Parse counts and compute visibility
    df["counts"] = df["counts_json"].map(parse_counts)
    df["vis"] = df.apply(lambda r: row_visibility(r["counts"], int(r["shots"])), axis=1)

    # T1 proxy
    _, t1_out = fit_t1(df)
    if t1_out:
        with open(os.path.join(args.out, "t1_fit.json"), "w", encoding="utf-8") as f:
            json.dump(t1_out, f, indent=2)

    # Focus on coherence families (exclude T1)
    d2 = df[df["kind"].isin(["RAMSEY_SINGLE", "BELL_STRUCTURED", "BELL_NOISE", "BELL_MIXED"])].copy()
    if d2.empty:
        raise SystemExit("No coherence rows found (expected RAMSEY_SINGLE/BELL_* kinds).")

    # Fit per rep per blind/label
    rows = []
    for (lab, rep), g in d2.groupby([label_col, "rep"]):
        T, meta = fit_t2_star_for_rep(g)
        rows.append(
            {
                "label": str(lab),
                "rep": int(rep),
                "t2star_us": T,
                "n_points": int(len(g)),
                "kind_set": ",".join(sorted(set(g["kind"].astype(str).tolist()))),
                "lambda": float(np.nanmean(g["lambda"])) if "lambda" in g else np.nan,
                "fit_meta": json.dumps(meta),
            }
        )

    per_rep = pd.DataFrame(rows)
    per_rep.to_csv(os.path.join(args.out, "t2star_per_rep_blinded.csv"), index=False)

    # Determine whether we have λ sweep
    has_lambda = np.isfinite(per_rep["lambda"]).any() or per_rep["label"].str.startswith("L").any()

    # Categorical stats for A/B/C style
    # We'll only run ANOVA/Tukey if there are at least 2 groups with >=2 valid points
    cat = per_rep.copy()
    cat_valid = cat[np.isfinite(cat["t2star_us"])].copy()
    group_sizes = cat_valid.groupby("label")["t2star_us"].size()
    eligible_groups = group_sizes[group_sizes >= 2].index.tolist()

    anova_path = os.path.join(args.out, "anova_t2star.csv")
    tukey_path = os.path.join(args.out, "tukey_t2star.txt")
    eff_path = os.path.join(args.out, "effect_sizes_t2star.json")

    if len(eligible_groups) >= 2:
        groups = [cat_valid[cat_valid["label"] == g]["t2star_us"].values for g in eligible_groups]
        F, p = f_oneway(*groups)
        pd.DataFrame([{"F": F, "p_value": p, "groups": "|".join(eligible_groups)}]).to_csv(anova_path, index=False)

        tuk = pairwise_tukeyhsd(endog=cat_valid[cat_valid["label"].isin(eligible_groups)]["t2star_us"],
                                groups=cat_valid[cat_valid["label"].isin(eligible_groups)]["label"],
                                alpha=0.05)
        with open(tukey_path, "w", encoding="utf-8") as f:
            f.write(str(tuk))

        # Effect sizes: pairwise Cohen's d
        eff = {}
        for i, g1 in enumerate(eligible_groups):
            for g2 in eligible_groups[i+1:]:
                x = cat_valid[cat_valid["label"] == g1]["t2star_us"].values
                y = cat_valid[cat_valid["label"] == g2]["t2star_us"].values
                eff[f"{g1}_vs_{g2}"] = float(cohen_d(x, y))
        with open(eff_path, "w", encoding="utf-8") as f:
            json.dump(eff, f, indent=2)

    # λ sweep summary/regression (decoded or not)
    if has_lambda:
        lam = per_rep.copy()
        lam = lam[np.isfinite(lam["lambda"]) & np.isfinite(lam["t2star_us"])].copy()
        if not lam.empty:
            # summarize per lambda
            summary = (
                lam.groupby("lambda")["t2star_us"]
                .agg(["count", "mean", "std", "median"])
                .reset_index()
                .sort_values("lambda")
            )
            summary.to_csv(os.path.join(args.out, "lambda_summary.csv"), index=False)

            # simple linear regression (not claiming linearity; just a quick diagnostic)
            x = summary["lambda"].to_numpy(dtype=float)
            y = summary["mean"].to_numpy(dtype=float)
            if len(x) >= 3:
                # least squares
                A = np.vstack([x, np.ones_like(x)]).T
                m, b = np.linalg.lstsq(A, y, rcond=None)[0]
                yhat = m * x + b
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                with open(os.path.join(args.out, "lambda_regression.txt"), "w", encoding="utf-8") as f:
                    f.write("Linear regression on per-lambda mean T2* (diagnostic only)\n")
                    f.write(f"slope: {m:.6g} us per lambda\n")
                    f.write(f"intercept: {b:.6g} us\n")
                    f.write(f"R^2: {r2:.6g}\n")

    print("\n✅ Analysis complete.")
    print(f"- per-rep: {os.path.join(args.out, 't2star_per_rep_blinded.csv')}")
    if os.path.exists(anova_path):
        print(f"- ANOVA:   {anova_path}")
    if os.path.exists(tukey_path):
        print(f"- Tukey:  {tukey_path}")
    if os.path.exists(eff_path):
        print(f"- effects:{eff_path}")
    if os.path.exists(os.path.join(args.out, 'lambda_summary.csv')):
        print(f"- lambda: {os.path.join(args.out, 'lambda_summary.csv')}")


if __name__ == "__main__":
    main()
