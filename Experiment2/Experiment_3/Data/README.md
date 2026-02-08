# Data — Experiment 3

This directory contains all numerical data used to generate the figures
reported in Experiment 3. The contents are organized to clearly separate
final figure-driving values from intermediate statistical validation and
raw execution artifacts.

---

## summary_tables/
Final numerical values used directly in plots and fits shown in
CORE_RESULTS. Each CSV corresponds unambiguously to a specific plotted
observable (e.g., amplitude, T2*, or ⟨XX⟩ as a function of λ) and is
sufficient to reproduce all reported figures.

---

## intermediate_analysis/
Intermediate statistical outputs retained to document analytical
provenance and validation. These include per-repetition T2* estimates,
ANOVA results, Tukey post-hoc tests, effect size calculations, and
regression diagnostics.

**Why T2\* statistics are categorized as intermediate analysis:**  
The files in this directory justify uncertainty estimates and support the
null conclusion that the decoherence rate T2* is independent of the
classical mixing parameter λ. While essential for statistical rigor,
these analyses are not themselves direct inputs to plotted figures.
For clarity and readability, only the final summarized quantities used
in figures are placed in `summary_tables/`, while full statistical
provenance is preserved here.

---

## interpretations/
Human-readable summaries describing the physical interpretation of each
analysis channel (e.g., X-basis vs Z-basis behavior). These files are
intended to aid understanding and communication but are not required for
numerical reproduction of results.

---

## raw_execution/
Execution-level data and metadata generated directly by simulator or
hardware runs, including raw measurement outputs and condition manifests.
These files provide reproducibility and provenance but are not required
to regenerate figures from the provided summary tables.

---

Raw shot-level data is not included in the core summary tables for
clarity; all reported figures are fully reproducible from the CSV files
in `summary_tables/`.
