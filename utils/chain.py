# Existing imports
from fx.delay import delay_fx
from fx.reverb import reverb_fx
from fx.overdrive import overdrive_fx
from fx.compressor import compressor_fx

# --- NEW IMPORTS ---
# Ensure these files exist in your 'fx' folder!
from fx.distortion import distortion_fx
from fx.tremolo import tremolo_fx
from fx.chorus import chorus_fx
from fx.equalizer import equalizer_fx

# Map the string names (from main.py) to the actual functions
EFFECTS = {
    # Dynamics & Drive
    "Compressor": lambda audio, sr, p: compressor_fx(audio, sr, **p),
    "Distortion": lambda audio, sr, p: distortion_fx(audio, sr, **p),
    "Overdrive":  lambda audio, sr, p: overdrive_fx(audio, sr, **p),
    
    # Filter
    "Equalizer":  lambda audio, sr, p: equalizer_fx(audio, sr, **p),
    
    # Modulation
    "Tremolo":    lambda audio, sr, p: tremolo_fx(audio, sr, **p),
    "Chorus":     lambda audio, sr, p: chorus_fx(audio, sr, **p),
    
    # Time & Space
    "Delay":      lambda audio, sr, p: delay_fx(audio, sr, **p),
    "Reverb":     lambda audio, sr, p: reverb_fx(audio, sr, **p),
}

def process_chain(audio, sr, chain):
    """
    Takes an audio signal and a list of (effect_name, params_dict).
    Passes the audio through each effect sequentially.
    """
    y = audio.copy()
    
    for fx_name, params in chain:
        func = EFFECTS.get(fx_name)
        if func:
            try:
                # **params unpacks the dictionary into arguments
                # e.g. tremolo_fx(y, sr, rate=5.0, depth=0.5, mix=1.0)
                y = func(y, sr, params)
            except TypeError as e:
                print(f"⚠️ Error processing {fx_name}: {e}")
                # If parameters don't match, return dry signal for this stage
                continue
        else:
            print(f"⚠️ Effect '{fx_name}' not found in EFFECTS dictionary.")
            
    return y