import numpy as np
def overdrive_fx(x, sr=None, gain=3.0, tone=0.2, mix=1.0):
    y = np.tanh(gain * x)
    if tone != 0.0:
        dx = np.concatenate([[0.0], np.diff(y)])
        y = (1 - tone) * y + tone * (y + 0.2 * dx)
    out = (1 - mix) * x + mix * y
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out
