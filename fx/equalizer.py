import numpy as np
from scipy.signal import butter, sosfilt

def equalizer_fx(x, sr=44100, low_gain=1.0, mid_gain=1.0, high_gain=1.0, mix=1.0):
    # Crossover frequencies
    low_cut = 400   # Hz
    high_cut = 3000 # Hz
    
    # 1. Design Filters (Butterworth 2nd order)
    sos_low = butter(2, low_cut, btype='low', fs=sr, output='sos')
    sos_high = butter(2, high_cut, btype='high', fs=sr, output='sos')
    # Bandpass is tricky to sum perfectly, so we subtract low and high from original to get mid
    
    # 2. Filter Separation
    low_band = sosfilt(sos_low, x)
    high_band = sosfilt(sos_high, x)
    mid_band = x - (low_band + high_band) # Simple way to get Mids
    
    # 3. Apply Gains
    y = (low_band * low_gain) + (mid_band * mid_gain) + (high_band * high_gain)
    
    # 4. Mix
    out = (1 - mix) * x + mix * y
    
    # Safety Check
    m = np.max(np.abs(out)) + 1e-9
    return out / m if m > 1.0 else out