from fx.delay import delay_fx
from fx.reverb import reverb_fx
from fx.overdrive import overdrive_fx
from fx.compressor import compressor_fx

EFFECTS = {
    "Compressor": lambda audio, sr, p: compressor_fx(audio, sr, **p),
    "Overdrive":  lambda audio, sr, p: overdrive_fx(audio, sr, **p),
    "Delay":      lambda audio, sr, p: delay_fx(audio, sr, **p),
    "Reverb":     lambda audio, sr, p: reverb_fx(audio, sr, **p),
}

def process_chain(audio, sr, chain):
    y = audio.copy()
    for fx_name, params in chain:
        func = EFFECTS.get(fx_name)
        if func:
            y = func(y, sr, params)
    return y
