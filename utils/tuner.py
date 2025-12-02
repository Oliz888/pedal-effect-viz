import numpy as np, librosa
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def estimate_f0(x, sr, frame_length=4096, hop=512, fmin=50, fmax=2000):
    f0 = librosa.yin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop)
    f0 = f0[np.isfinite(f0)]
    if len(f0)==0: return np.nan
    return float(np.median(f0))
def f0_to_note_cents(f0):
    if not np.isfinite(f0) or f0<=0: return "—", 0, f0
    midi = 69 + 12*np.log2(f0/440.0)
    mround = int(np.round(midi))
    name = f"{NOTE_NAMES[mround%12]}{(mround//12)-1}"
    ref = 440.0*(2**((mround-69)/12))
    cents = int(np.round(1200*np.log2(f0/ref)))
    return name, cents, f0
