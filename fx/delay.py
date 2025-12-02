import numpy as np
def delay_fx(x, sr, delay_ms=300, feedback=0.4, mix=0.3):
    D = max(1, int(sr * delay_ms / 1000))
    y = np.zeros_like(x)
    for n in range(len(x)):
        wet = y[n - D] * feedback if n >= D else 0.0
        y[n] = x[n] + wet
    out = (1 - mix) * x + mix * y
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out
