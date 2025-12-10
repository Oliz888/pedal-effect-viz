import numpy as np

def distortion_fx(x, sr=44100, drive=10.0, threshold=0.3, mix=1.0):
    # 1. Pre-gain (Drive)
    wet = x * drive
    
    # 2. Hard Clipping (The "Chop")
    # Anything above threshold becomes threshold
    wet = np.clip(wet, -threshold, threshold)
    
    # 3. Makeup Gain (Optional: normalize volume after clipping)
    # This helps keep volume steady so it doesn't get too quiet
    wet = wet / threshold * 0.5 

    # 4. Mix
    out = (1 - mix) * x + mix * wet
    
    # Safety Check
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out