import numpy as np

def chorus_fx(x, sr=44100, rate=1.5, depth_ms=2.0, mix=0.5):
    N = len(x)
    t = np.arange(N) / sr
    
    # 1. Calculate variable delay in samples
    # Base delay 15ms + oscillating depth
    base_delay_ms = 15.0 
    mod_ms = depth_ms * np.sin(2 * np.pi * rate * t)
    total_delay_samples = (base_delay_ms + mod_ms) * (sr / 1000.0)
    
    # 2. Vectorized Linear Interpolation
    # "Where was the signal X samples ago?"
    read_idx = np.arange(N) - total_delay_samples
    
    # Handle edges (pad with 0)
    read_idx = np.clip(read_idx, 0, N - 2)
    
    idx_floor = read_idx.astype(int)
    idx_ceil = idx_floor + 1
    frac = read_idx - idx_floor
    
    # Interpolate between sample A and sample B
    wet = (1 - frac) * x[idx_floor] + frac * x[idx_ceil]
    
    # 3. Mix
    out = (1 - mix) * x + mix * wet
    
    # Safety Check
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out