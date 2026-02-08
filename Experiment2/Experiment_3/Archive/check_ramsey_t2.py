import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv('results_lambda_sweep_clean/lambda_sweep_clean_1792032905.csv')
ramsey = df[df['kind']=='RAMSEY_SINGLE'].copy()
ramsey['vis'] = ramsey['p00'] - ramsey['p11']

def exp_decay(t, a, T, c):
    return a*np.exp(-t/T)+c

print("RAMSEY T2* per rep:")
for rep in ramsey['rep'].unique()[:5]:
    g = ramsey[ramsey['rep']==rep].sort_values('delay_us')
    try:
        popt, _ = curve_fit(
            exp_decay,
            g['delay_us'],
            g['vis'],
            p0=[1, 80, 0],
            bounds=([-2, 1, -1], [2, 500, 1])
        )
        print(f"Rep {rep}: T2* = {popt[1]:.1f} us")
    except Exception as e:
        print(f"Rep {rep}: Fit failed - {e}")

print(f"\nMean RAMSEY visibility at t=0: {ramsey[ramsey['delay_us']<1]['vis'].mean():.3f}")
print(f"Mean RAMSEY visibility at t=200: {ramsey[ramsey['delay_us']>190]['vis'].mean():.3f}")
