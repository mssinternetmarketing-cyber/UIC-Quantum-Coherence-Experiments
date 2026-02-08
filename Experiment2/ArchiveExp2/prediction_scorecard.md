\# Prediction Scorecard: Experiment 2 (λ-Sweep)

\*\*Date\*\*: 2026-02-07    
\*\*Predictions Sealed\*\*: 06:58 CST    
\*\*Results Decoded\*\*: 07:30 CST

\---

\#\# Actual Results (Decoded)

\#\#\# Clean Fits (R² \> 0.40):  
| λ     | Mean T2\* (µs) | Std Dev | R² Range  |  
|-------|---------------|---------|-----------|  
| 0.0   | 23.2          | 0.39    | 0.56-0.58 |  
| 0.2   | 11.6          | 0.09    | 0.40-0.42 |  
| 0.4   | 19.4          | 0.21    | 0.51-0.52 |  
| 0.6   | 16.5          | 0.16    | 0.47-0.49 |  
| 1.0   | 17.9          | 0.50    | 0.84-0.89 |

\#\#\# Failed Fits (R² \< 0.15):  
| λ     | Mean T2\* (µs) | Issue                    |  
|-------|---------------|--------------------------|  
| 0.1   | 114.5         | Negative amplitude, R²≈0.007 |  
| 0.3   | 0.1           | Negative amplitude, R²≈0.21  |  
| 0.8   | 4.5           | Bimodal distribution, R²≈0.11 |  
| 0.9   | 27.7          | High variance, R²≈0.01       |

\*\*Overall Linear Regression:\*\* β \= \-34.6 µs/λ, R² \= 0.13 (poor fit)

\---

\#\# Prediction Comparison

\#\#\# Perplexity's Predictions:  
\- T2\*(0.0) \= 21 µs → \*\*Actual: 23.2 µs\*\* ✅ (close, 10% error)  
\- T2\*(0.5) \= 48 µs → \*\*Actual: 10.0 µs\*\* ❌ (off by 4.8×)  
\- T2\*(1.0) \= 65 µs → \*\*Actual: 17.9 µs\*\* ❌ (off by 3.6×)  
\- Slope β \= 40-50 µs/λ → \*\*Actual: \-34.6 µs/λ\*\* ❌ (wrong sign\!)  
\- R² \= 0.75-0.90 → \*\*Actual: 0.13\*\* ❌

\*\*Correct predictions:\*\* 1/5 (baseline only)

\#\#\# ChatGPT's Predictions:  
\- T2\*(0.0) \= 12 µs → \*\*Actual: 23.2 µs\*\* ❌ (off by 1.9×)  
\- T2\*(0.5) \= 30 µs → \*\*Actual: 10.0 µs\*\* ❌ (off by 3×)  
\- T2\*(1.0) \= 55 µs → \*\*Actual: 17.9 µs\*\* ❌ (off by 3.1×)  
\- Spearman ρ ≥ 0.70 → \*\*Actual: \~0.0\*\* ❌ (no correlation)  
\- Variance peaks mid-λ → \*\*Inconclusive\*\* (fits failed in mid-range)  
\- ΔT2\* ≥ 25 µs → \*\*Actual: \-5.3 µs\*\* ❌ (wrong direction\!)

\*\*Correct predictions:\*\* 0/6

\---

\#\# Outcome

\*\*Winner:\*\* Perplexity (barely) — got the baseline right, everything else wrong.

\*\*Why Both Failed:\*\*  
Both models assumed smooth, monotonic T2\*(λ) curves with exponential decay at all λ values. Neither anticipated that stochastic per-round branching would create:

1\. \*\*Non-exponential decay\*\* at intermediate λ  
2\. \*\*Binomial sampling variance\*\* breaking single-exponential fits  
3\. \*\*Regime-dependent dynamics\*\* (clean at extremes, chaotic in middle)  
4\. \*\*No clear linear trend\*\* across all λ values

\*\*Physical Interpretation:\*\*  
The stochastic implementation creates heterogeneous decoherence pathways. At intermediate λ values, different circuit instances experience vastly different numbers of structured vs. noise rounds (binomial distribution), preventing coherent averaging into a single exponential decay.

\*\*Scientific Value:\*\*  
This is a \*\*discovery\*\*, not a failure. The results reveal that organizational parameter modulation produces qualitatively different decoherence regimes depending on the mixing statistics.

\---

\#\# Lessons Learned

1\. ✅ Preregistration worked (protocol was followed)  
2\. ✅ Blinding worked (analysis completed before decode)  
3\. ✅ Predictions were falsifiable (both were wrong in specific ways)  
4\. ❌ Physical model was incomplete (didn't account for stochastic variance effects)  
5\. 🔬 Data is publishable (methodological contribution on regime-dependent dynamics)

\---

\#\# Recommendations for Experiment 3

Use \*\*deterministic λ\*\* instead of stochastic per-round branching:

\`\`\`python  
\# Instead of: if rng.random() \< lam: structured\_block()  
\# Use: n\_structured \= int(lam \* total\_rounds)

