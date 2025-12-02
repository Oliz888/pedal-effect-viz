import numpy as np, soundfile as sf
from scipy.signal import fftconvolve
def reverb_fx(x, sr, ir_path='assets/impulse_responses/room.wav', mix=0.3, pre_delay_ms=0.0):
    ir, ir_sr = sf.read(ir_path)
    if ir.ndim > 1: ir = ir.mean(axis=1)
    if ir_sr != sr:
        ratio = sr / ir_sr
        idx = (np.arange(int(len(ir)*ratio)) / ratio).astype(int)
        idx[idx >= len(ir)] = len(ir)-1
        ir = ir[idx]
    if pre_delay_ms > 0:
        zeros = np.zeros(int(sr*pre_delay_ms/1000))
        ir = np.concatenate([zeros, ir])
    y = fftconvolve(x, ir, mode="full")[:len(x)]
    out = (1 - mix) * x + mix * y
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out
