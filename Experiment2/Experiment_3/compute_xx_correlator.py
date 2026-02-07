import pandas as pd
import numpy as np

df = pd.read_csv('results_lambda_correct/lambda_final_138280232.csv')

qx = df[(df['kind'] == 'QUANTUM_XBASIS') & (df['rounds'] == 5) & (df['delay_us'] == 0)]
cx = df[(df['kind'] == 'CLASSICAL_XBASIS') & (df['rounds'] == 5) & (df['delay_us'] == 0)]

lambda_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print('X-BASIS PROBABILITIES AT t=0:')
print('=' * 70)

for lam in lambda_vals:
    # Get quantum and classical averages
    q_avg = qx[['p00', 'p01', 'p10', 'p11']].mean()
    c_avg = cx[['p00', 'p01', 'p10', 'p11']].mean()

    # Mix probabilities
    p_mixed = (1 - lam) * q_avg + lam * c_avg

    # Compute <XX> = P++ + P-- - P+- - P-+
    xx = p_mixed['p00'] + p_mixed['p11'] - p_mixed['p01'] - p_mixed['p10']

    print(f"λ={lam:.1f}: P++={p_mixed['p00']:.4f}, P+-={p_mixed['p01']:.4f}, "
          f"P-+={p_mixed['p10']:.4f}, P--={p_mixed['p11']:.4f}, ⟨XX⟩={xx:.4f}")
