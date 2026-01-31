import numpy as np
import pandas as pd
from portfolio_backtester.numba_kernels import drifting_weights_returns_kernel

def pandas_drifting_returns(target_weights, rets, mask):
    T, N = target_weights.shape
    out = np.zeros(T)
    w = target_weights[0].copy()
    
    for i in range(1, T):
        # Rebalance if target weights changed
        if not np.allclose(target_weights[i], target_weights[i-1], atol=1e-9):
            w = target_weights[i].copy()
            
        # Gross return
        gross = np.sum(w * rets[i])
        out[i] = gross
        
        # Drift
        denom = 1.0 + gross
        if denom > 1e-9:
            w = w * (1.0 + rets[i]) / denom
            
    return out

# Test Data
T, N = 100, 10
np.random.seed(42)
rets = np.random.normal(0.001, 0.02, (T, N))
weights = np.zeros((T, N))
for i in range(0, T, 10): # Rebalance every 10 days
    weights[i:i+10] = 0.1
mask = np.ones((T, N), dtype=bool)

# Numba Result
numba_out = drifting_weights_returns_kernel(weights.astype(np.float32), rets.astype(np.float32), mask)

# Pandas Result
pd_out = pandas_drifting_returns(weights, rets, mask)

diff = np.abs(numba_out - pd_out).max()
print(f"Max difference: {diff}")
if diff < 1e-5:
    print("Numba kernel verified successfully!")
else:
    print("Numba kernel discrepancy detected!")
