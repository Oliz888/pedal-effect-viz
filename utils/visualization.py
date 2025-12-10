import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display

def wave_compare(original, processed, sr):
    """
    Standard Time-Domain comparison.
    Best for: Seeing LFO movement (Tremolo) or Gross Dynamics (Compression).
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    librosa.display.waveshow(original, sr=sr, ax=ax[0], alpha=0.8) # Added alpha for density
    ax[0].set_title("Original")
    librosa.display.waveshow(processed, sr=sr, ax=ax[1], alpha=0.8)
    ax[1].set_title("Processed")
    fig.tight_layout()
    return fig

def spec_compare(original, processed, sr, n_fft=2048, hop=512):
    """
    Frequency-Domain comparison.
    Best for: Seeing Harmonic Distortion (Vertical lines) or Filter shapes (EQ).
    """
    So = np.abs(librosa.stft(original, n_fft=n_fft, hop_length=hop))
    Sp = np.abs(librosa.stft(processed, n_fft=n_fft, hop_length=hop))
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Using vmin/vmax ensures the colors scales match for fair comparison
    img1 = librosa.display.specshow(librosa.amplitude_to_db(So, ref=np.max), 
                             sr=sr, hop_length=hop, y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set_title("Original Spectrogram")
    fig.colorbar(img1, ax=ax[0], format="%+2.0f dB")
    
    img2 = librosa.display.specshow(librosa.amplitude_to_db(Sp, ref=np.max), 
                             sr=sr, hop_length=hop, y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set_title("Processed Spectrogram")
    fig.colorbar(img2, ax=ax[1], format="%+2.0f dB")
    
    fig.tight_layout()
    return fig

def frame_slider_plot(orig, proc, sr, frame_ms=120, frame_index=0):
    """
    Micro-Time comparison.
    Best for: Seeing the actual shape of Distortion (Square waves) or Phase shift.
    """
    frame_len = int(sr * frame_ms / 1000)
    total_frames = max(1, len(orig)//frame_len)
    start = frame_index * frame_len; end = min(start+frame_len, len(orig))
    
    o_seg = orig[start:end]; p_seg = proc[start:end]
    
    if len(o_seg) < frame_len:
        o_seg = np.pad(o_seg, (0, frame_len-len(o_seg)))
        p_seg = np.pad(p_seg, (0, frame_len-len(p_seg)))
        
    t = np.linspace(0, frame_ms/1000.0, frame_len)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax[0].plot(t, o_seg)
    ax[0].set_title(f"Original (Frame {frame_index+1}/{total_frames})")
    ax[0].grid(True, alpha=0.3) # Added Grid
    
    ax[1].plot(t, p_seg)
    ax[1].set_title("Processed")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True, alpha=0.3) # Added Grid
    
    fig.tight_layout()
    return fig, total_frames

def transfer_compare(original, processed, downsample=10):
    """
    *NEW* Input vs Output Scatter Plot (Phase Portrait).
    Best for: Visualizing the 'Math' (Linearity vs Non-linearity).
    Args:
        downsample: Integers to skip samples. Plotting 44100 dots is slow/messy.
                    Skip every 10th sample for speed.
    """
    # Downsample for performance and clarity
    x = original[::downsample]
    y = processed[::downsample]
    
    fig, ax = plt.subplots(figsize=(6, 6)) # Square aspect ratio is standard for this
    
    # 1. Plot the "System State" (The dots)
    # Alpha=0.1 makes it look like a "cloud" or "heat map"
    ax.scatter(x, y, s=1, alpha=0.1, color='blue', label='Signal State')
    
    # 2. Plot the Reference Line (Perfectly Clean)
    # Helps audience see deviation from "normal"
    lims = [min(min(x), min(y)), max(max(x), max(y))]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Linear Reference (y=x)')
    
    ax.set_title("Input/Output Transfer Function")
    ax.set_xlabel("Input Amplitude (x[n])")
    ax.set_ylabel("Processed Amplitude (y[n])")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    return fig