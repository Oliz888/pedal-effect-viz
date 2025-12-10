import numpy as np

def tremolo_fx(x, sr=44100, rate=5.0, depth=0.5, mix=1.0):
    # 1. Create the LFO (Low Frequency Oscillator)
    t = np.arange(len(x)) / sr
    lfo = 0.5 * (1.0 + np.sin(2 * np.pi * rate * t)) # Oscillates 0 to 1
    
    # 2. Apply volume modulation
    # Depth 0.0 = no change, Depth 1.0 = silence to full volume
    gain_mod = (1.0 - depth) + (depth * lfo)
    wet = x * gain_mod
    
    # 3. Mix and Output
    out = (1 - mix) * x + mix * wet
    
    # Safety Check (Prevent Clipping)
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out