import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display

def wave_compare(original, processed, sr):
    fig, ax = plt.subplots(2,1, figsize=(10,4), sharex=True)
    librosa.display.waveshow(original, sr=sr, ax=ax[0])
    ax[0].set_title("Original")
    librosa.display.waveshow(processed, sr=sr, ax=ax[1])
    ax[1].set_title("Processed")
    fig.tight_layout()
    return fig

def spec_compare(original, processed, sr, n_fft=2048, hop=512):
    So = np.abs(librosa.stft(original, n_fft=n_fft, hop_length=hop))
    Sp = np.abs(librosa.stft(processed, n_fft=n_fft, hop_length=hop))
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(So, ref=np.max), sr=sr, hop_length=hop, y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title("Original Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(Sp, ref=np.max), sr=sr, hop_length=hop, y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title("Processed Spectrogram")
    fig.tight_layout()
    return fig

def frame_slider_plot(orig, proc, sr, frame_ms=120, frame_index=0):
    frame_len = int(sr * frame_ms / 1000)
    total_frames = max(1, len(orig)//frame_len)
    start = frame_index * frame_len; end = min(start+frame_len, len(orig))
    o_seg = orig[start:end]; p_seg = proc[start:end]
    if len(o_seg) < frame_len:
        o_seg = np.pad(o_seg, (0, frame_len-len(o_seg))); p_seg = np.pad(p_seg, (0, frame_len-len(p_seg)))
    t = np.linspace(0, frame_ms/1000.0, frame_len)
    fig, ax = plt.subplots(2,1, figsize=(10,4), sharex=True)
    ax[0].plot(t, o_seg); ax[0].set_title(f"Original (Frame {frame_index+1}/{total_frames})")
    ax[1].plot(t, p_seg); ax[1].set_title("Processed"); ax[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig, total_frames
