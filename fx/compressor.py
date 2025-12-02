import numpy as np
def compressor_fx(x, sr=None, threshold=0.4, ratio=4.0, makeup=1.0, mix=1.0):
    mag = np.abs(x); sign = np.sign(x)
    under = mag < threshold; over = ~under
    y = np.empty_like(x)
    y[under] = x[under]
    y[over] = sign[over] * (threshold + (mag[over]-threshold)/ratio)
    y *= makeup
    out = (1 - mix) * x + mix * y
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out
