"""
⟨XX⟩ Correlator Kill Shot
==========================

Creates matching plot for XX correlator with same style as amplitude killshot.
"""

import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Data from compute_xx_correlator.py
lambda_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
xx_vals = np.array([0.7826, 0.7049, 0.6272, 0.5495, 0.4718, 0.3942, 0.3165, 0.2388, 0.1611, 0.0834, 0.0057])

# Estimate uncertainties (from shot noise)
# For 2000 shots, σ ≈ sqrt(p(1-p)/N) for each outcome
# Conservative estimate: ~0.01 for correlator
xx_err = np.full_like(xx_vals, 0.01)

# Normalize
XX_0 = xx_vals[0]
XX_norm = xx_vals / XX_0
XX_norm_err = xx_err / XX_0

print("="*70)
print("⟨XX⟩ CORRELATOR ANALYSIS")
print("="*70)
print(f"\n⟨XX⟩(0) = {XX_0:.6f}\n")
print("Normalized values:")
for lam, xx_norm in zip(lambda_vals, XX_norm):
    print(f"λ = {lam:.1f}: ⟨XX⟩_norm = {xx_norm:.6f}")

# FIT 1: Unconstrained linear
def linear(x, m, b):
    return m * x + b

popt, pcov = curve_fit(linear, lambda_vals, XX_norm, sigma=XX_norm_err, absolute_sigma=True)
m, b = popt
m_err, b_err = np.sqrt(np.diag(pcov))

XX_fit = linear(lambda_vals, m, b)
ss_res = np.sum((XX_norm - XX_fit)**2)
ss_tot = np.sum((XX_norm - np.mean(XX_norm))**2)
r2 = 1 - ss_res/ss_tot

residuals = XX_norm - XX_fit
rms_residual = np.sqrt(np.mean(residuals**2))

print("\n" + "="*70)
print("FIT 1: UNCONSTRAINED LINEAR")
print("="*70)
print(f"\n⟨XX⟩_norm = m*λ + b")
print(f"  m = {m:.6f} ± {m_err:.6f}  (target: -1)")
print(f"  b = {b:.6f} ± {b_err:.6f}  (target: 1)")
print(f"  R² = {r2:.6f}  (target: 1)")
print(f"  RMS residual = {rms_residual:.6f}")

# FIT 2: Constrained theory
XX_theory = 1 - lambda_vals
residuals_theory = XX_norm - XX_theory
rms_theory = np.sqrt(np.mean(residuals_theory**2))

chi2 = np.sum(((XX_norm - XX_theory) / XX_norm_err)**2)
dof = len(lambda_vals)
chi2_per_dof = chi2 / dof

print("\n" + "="*70)
print("FIT 2: CONSTRAINED THEORY")
print("="*70)
print(f"\n⟨XX⟩_norm = 1 - λ (no free parameters)")
print(f"  χ² = {chi2:.4f}")
print(f"  dof = {dof}")
print(f"  χ²/dof = {chi2_per_dof:.4f}")
print(f"  RMS residual = {rms_theory:.6f}")

# PLOT (matching amplitude killshot style)
fig = plt.figure(figsize=(10, 8))

# Main plot
ax1 = plt.subplot(2, 1, 1)

ax1.errorbar(lambda_vals, XX_norm, yerr=XX_norm_err, fmt='o',
             markersize=8, capsize=5, linewidth=2, color='blue',
             label='Measured ⟨XX⟩', zorder=3)

ax1.plot(lambda_vals, XX_theory, 'r--', linewidth=3,
         label=r'Theory: $\langle XX \rangle_{norm} = 1-\lambda$', zorder=2)

lambda_dense = np.linspace(0, 1, 100)
ax1.plot(lambda_dense, linear(lambda_dense, m, b), 'g:', linewidth=2,
         label=f'Linear fit: m={m:.4f}, b={b:.4f}', zorder=1)

ax1.set_xlabel(r'$\lambda$ (Classical Mixing Parameter)', fontsize=14, fontweight='bold')
ax1.set_ylabel(r'$\langle XX \rangle(\lambda) / \langle XX \rangle(0)$', fontsize=14, fontweight='bold')
ax1.set_title('Normalized XX Correlator vs Classical Mixing', fontsize=16, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Text box
textstr = f'R² = {r2:.5f}\nRMS = {rms_residual:.5f}\nχ²/dof = {chi2_per_dof:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.15, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# Residual plot
ax2 = plt.subplot(2, 1, 2)
ax2.errorbar(lambda_vals, residuals_theory, yerr=XX_norm_err, fmt='o',
             markersize=8, capsize=5, linewidth=2, color='purple')
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel(r'$\lambda$', fontsize=14, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=14, fontweight='bold')
ax2.set_title('Residuals: Measured - Theory', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

os.makedirs('killshot_analysis', exist_ok=True)
plt.savefig('killshot_analysis/xx_correlator_killshot.png', dpi=200)
print(f"\n✅ Plot saved: killshot_analysis/xx_correlator_killshot.png\n")
