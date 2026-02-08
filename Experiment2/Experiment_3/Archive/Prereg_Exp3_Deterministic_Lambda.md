\# Preregistration: Experiment 3 — Deterministic λ-Sweep

\*\*Date\*\*: 2026-02-07    
\*\*Author\*\*: Monet (with Perplexity AI)    
\*\*Status\*\*: Locked before execution

\---

\#\# 1\. Research Question

\*\*Does deterministic (vs. stochastic) implementation of organizational parameter λ   
produce clean, interpretable T2\*(λ) scaling in Bell state coherence measurements?\*\*

\---

\#\# 2\. Background & Motivation

\#\#\# Experiment 2 Results:  
\- Stochastic per-round λ branching produced non-exponential dynamics at intermediate λ  
\- Exponential fits failed (R² \< 0.15) for λ ∈ {0.1, 0.3, 0.8, 0.9}  
\- Clean fits only at extremes (λ \= 0.0, 0.2, 0.4, 0.6, 1.0)  
\- No clear linear T2\*(λ) relationship (overall R² \= 0.13)

\#\#\# Hypothesis:  
Binomial sampling variance from stochastic branching broke exponential fitting.   
Deterministic implementation should restore clean fits at all λ values.

\#\#\# Secondary Question:  
Does structured feedback (CX-RZ-CX) actually protect coherence better than random   
Pauli noise, or is the 2Q depolarizing noise overwhelming organizational benefits?

\---

\#\# 3\. Experimental Design

\#\#\# Circuit Architecture:  
\- \*\*Initial state\*\*: Bell state |Φ+⟩ \= (|00⟩ \+ |11⟩)/√2  
\- \*\*Feedback mechanism\*\*: Deterministic organizational blocks  
\- \*\*Measurement\*\*: Parity decay via computational basis measurement

\#\#\# Deterministic λ Implementation:

For each λ value and feedback round total (4 rounds):  
\`\`\`python  
n\_structured \= int(λ \* 4\)  \# Number of structured blocks  
n\_noise \= 4 \- n\_structured  \# Number of noise blocks

\# Apply blocks deterministically (not stochastically)  
for i in range(n\_structured):  
    apply\_structured\_block()  \# CX-RZ-CX \+ 2Q depol  
      
for i in range(n\_noise):  
    apply\_noise\_block()  \# Random Pauli \+ 1Q depol

