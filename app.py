import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Custom modules
from utils.chain import process_chain
# [CHANGE 1] Added transfer_compare to imports
from utils.visualization import wave_compare, spec_compare, frame_slider_plot, transfer_compare
from utils.tuner import estimate_f0, f0_to_note_cents

# -----------------------------
# 0. Page Config
# -----------------------------
st.set_page_config(page_title="Audio FX Explorer Playground", layout="wide")
st.title("🎛️ Pedal Effect Explorer")

# -----------------------------
# 1. Input Section
# -----------------------------
uploaded = st.sidebar.file_uploader("Upload WAV/MP3", type=["wav", "mp3"], key="upl")
use_demo = st.sidebar.checkbox("Use demo sine (440 Hz)", value=not uploaded, key="demo")

# Load Audio
if use_demo:
    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
else:
    if uploaded:
        y, sr = librosa.load(uploaded, sr=44100, mono=True)
    else:
        sr = 44100
        y = np.zeros(int(sr * 2), dtype=np.float32)

# -----------------------------
# 2. Pedalboard Visuals Helper
# -----------------------------
st.subheader("Pedalboard")

def render_pedal(title, bg, led_on, config):
    path = Path("components/pedal_canvas.html")
    if path.exists():
        html = path.read_text()
        html = (
            html.replace("{{title}}", title)
            .replace("{{bg}}", bg)
            .replace("{{led}}", "on" if led_on else "")
            .replace("{{config}}", str(config).replace("'", '"'))
        )
        st.components.v1.html(html, height=380)
    else:
        st.error("⚠️ components/pedal_canvas.html not found.")

row1 = st.columns(4, gap="medium")
row2 = st.columns(4, gap="medium")

# -----------------------------
# 3. ROW 1: Dynamics & Tone
# -----------------------------

# --- [1] Compressor ---
with row1[0]:
    st.markdown("**1. Compressor**")
    comp_on = st.checkbox("On", value=False, key="comp_on")
    T = st.slider("Threshold", 0.05, 0.9, 0.4, key="comp_T")
    R = st.slider("Ratio", 1.0, 20.0, 4.0, key="comp_R")
    MIXC = st.slider("Mix", 0.0, 1.0, 1.0, key="comp_mix")

    render_pedal("COMP", "#ff4d4d", comp_on, {
        "time": min((R - 1) / 19, 1.0), "fb": T, "mix": MIXC,
        "labels": {"time": "RATIO", "fb": "THRESH", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""}
    })
    
    with st.expander("Theory"):
        st.latex(r"y = \begin{cases}x, & |x|<T\\ \operatorname{sign}(x)\left(T+\frac{|x|-T}{R}\right), & |x|\ge T\end{cases}")
        st.markdown(
            f"""**T = {T:.2f}** – amplitude threshold.  
**R = {R:.1f}** – higher means stronger compression.  
**Mix = {MIXC:.2f}** – dry/wet blend."""
        )

# --- [2] Distortion (Hard Clip) ---
with row1[1]:
    st.markdown("**2. Distortion**")
    dist_on = st.checkbox("On", value=False, key="dist_on")
    DRIVE = st.slider("Drive", 1.0, 50.0, 10.0, key="dist_drive")
    D_THRESH = st.slider("Ceiling", 0.1, 1.0, 0.3, key="dist_thresh")
    MIXDST = st.slider("Mix", 0.0, 1.0, 1.0, key="dist_mix")

    render_pedal("DISTORT", "#cc0000", dist_on, {
        "time": DRIVE/50.0, "fb": D_THRESH, "mix": MIXDST,
        "labels": {"time": "DRIVE", "fb": "CEIL", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""}
    })

    with st.expander("Theory"):
        st.caption("Hard Clipping")
        st.latex(r"y = \text{clip}(x \cdot \text{Drive}, -T, T)")
        st.markdown(
            f"""**Drive = {DRIVE:.1f}** – input gain multiplier.  
**Ceiling = {D_THRESH:.2f}** – absolute amplitude limit.  
**Mix = {MIXDST:.2f}** – dry/wet blend."""
        )

# --- [3] Overdrive (Soft Clip) ---
with row1[2]:
    st.markdown("**3. Overdrive**")
    drv_on = st.checkbox("On", value=False, key="drv_on")
    G = st.slider("Gain", 1.0, 12.0, 3.0, key="drv_G")
    TONE = st.slider("Tone", 0.0, 1.0, 0.2, key="drv_tone")
    MIXO = st.slider("Mix", 0.0, 1.0, 1.0, key="drv_mix")

    render_pedal("DRIVE", "#2aa6ff", drv_on, {
        "time": (G - 1) / 11, "fb": TONE, "mix": MIXO,
        "labels": {"time": "GAIN", "fb": "TONE", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""}
    })

    with st.expander("Theory"):
        st.caption("Soft Clipping / Tanh")
        st.latex(r"y = (1-m)x + m\tanh(G \cdot x)")
        st.markdown(
            f"""**G = {G:.2f}** – pre-gain into the waveshaper.  
**Tone = {TONE:.2f}** – brightness emphasis.  
**Mix = {MIXO:.2f}** – dry/wet blend."""
        )

# --- [4] Equalizer (3-Band) ---
with row1[3]:
    st.markdown("**4. Equalizer**")
    eq_on = st.checkbox("On", value=False, key="eq_on")
    LOW = st.slider("Low", 0.0, 2.0, 1.0, key="eq_low")
    MID = st.slider("Mid", 0.0, 2.0, 1.0, key="eq_mid")
    HI = st.slider("High", 0.0, 2.0, 1.0, key="eq_hi")
    
    render_pedal("EQ", "#dddddd", eq_on, {
        "time": LOW/2.0, "fb": MID/2.0, "mix": HI/2.0,
        "labels": {"time": "LOW", "fb": "MID", "mix": "HIGH"},
        "values": {"time": "", "fb": "", "mix": ""}
    })
    
    with st.expander("Theory"):
        st.caption("Parallel Biquad Filters")
        st.latex(r"y = G_L x_L + G_M x_M + G_H x_H")
        st.markdown(
            f"""**Low = {LOW:.2f}** – gain for bass frequencies.  
**Mid = {MID:.2f}** – gain for middle frequencies.  
**High = {HI:.2f}** – gain for treble frequencies."""
        )

# -----------------------------
# 4. ROW 2: Modulation & Time
# -----------------------------

# --- [5] Tremolo ---
with row2[0]:
    st.markdown("**5. Tremolo**")
    trem_on = st.checkbox("On", value=False, key="trem_on")
    T_RATE = st.slider("Rate (Hz)", 0.5, 10.0, 5.0, key="t_rate")
    T_DEPTH = st.slider("Depth", 0.0, 1.0, 0.5, key="t_depth")
    MIXT = st.slider("Mix", 0.0, 1.0, 1.0, key="t_mix")

    render_pedal("TREMOLO", "#e6e600", trem_on, {
        "time": T_RATE/10.0, "fb": T_DEPTH, "mix": MIXT,
        "labels": {"time": "RATE", "fb": "DEPTH", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""}
    })

    with st.expander("Theory"):
        st.caption("Amplitude Modulation (AM)")
        st.latex(r"y = x \cdot (1 - D + D\sin(2\pi f n))")
        st.markdown(
            f"""**Rate = {T_RATE:.1f} Hz** – LFO speed.  
**Depth = {T_DEPTH:.2f}** – intensity of volume modulation.  
**Mix = {MIXT:.2f}** – dry/wet blend."""
        )

# --- [6] Chorus ---
with row2[1]:
    st.markdown("**6. Chorus**")
    chor_on = st.checkbox("On", value=False, key="chor_on")
    C_RATE = st.slider("Rate (Hz)", 0.1, 5.0, 1.5, key="c_rate")
    C_DEPTH = st.slider("Depth (ms)", 0.1, 5.0, 2.0, key="c_depth")
    MIXC_CH = st.slider("Mix", 0.0, 1.0, 0.5, key="c_mix")

    render_pedal("CHORUS", "#b366ff", chor_on, {
        "time": C_RATE/5.0, "fb": C_DEPTH/5.0, "mix": MIXC_CH,
        "labels": {"time": "RATE", "fb": "DEPTH", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""}
    })

    with st.expander("Theory"):
        st.caption("Modulated Delay Line")
        st.latex(r"y[n] = x[n] + x[n - D(t)]")
        st.markdown(
            f"""**Rate = {C_RATE:.1f} Hz** – LFO oscillation speed.  
**Depth = {C_DEPTH:.1f} ms** – delay modulation range.  
**Mix = {MIXC_CH:.2f}** – dry/wet blend."""
        )

# --- [7] Delay ---
with row2[2]:
    st.markdown("**7. Delay**")
    d_on = st.checkbox("On", value=False, key="d_on")
    DMS = st.slider("Time (ms)", 50, 800, 300, key="d_ms")
    FB = st.slider("Feedback", 0.0, 0.95, 0.4, key="d_fb")
    MIXD = st.slider("Mix", 0.0, 1.0, 0.3, key="d_mix")

    render_pedal("DELAY", "#16c2b0", d_on, {
        "time": (DMS - 50) / 750, "fb": FB, "mix": MIXD,
        "labels": {"time": "TIME", "fb": "FEED", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""}
    })
    
    with st.expander("Theory"):
        st.latex(r"y[n] = x[n] + \alpha y[n-D]")
        st.markdown(
            f"""**α = {FB:.2f}** – echo persistence.  
**D = {DMS} ms** – delay spacing.  
**Mix = {MIXD:.2f}** – wet/dry balance."""
        )

# --- [8] Reverb ---
with row2[3]:
    st.markdown("**8. Reverb**")
    r_on = st.checkbox("On", value=False, key="r_on")
    PRD = st.slider("Pre-delay", 0, 200, 0, key="r_pre")
    ROOM = st.slider("Size", 0.5, 2.0, 1.0, key="r_room")
    MIXR = st.slider("Mix", 0.0, 1.0, 0.3, key="r_mix")

    render_pedal("REVERB", "#6aa5ff", r_on, {
        "time": PRD / 200, "fb": ROOM / 2.0, "mix": MIXR,
        "labels": {"time": "PRE", "fb": "SIZE", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""}
    })

    with st.expander("Theory"):
        st.latex(r"y = x * h_{room}")
        st.markdown(
            f"""**Room Size = {ROOM:.2f}×** – stretches impulse response.  
**Pre-delay = {PRD} ms** – gap before reflections.  
**Mix = {MIXR:.2f}** – wet level."""
        )

# -----------------------------
# 5. Signal Chain Processing
# -----------------------------
chain = []

if comp_on:
    chain.append(("Compressor", {"threshold": T, "ratio": R, "makeup": 1.0, "mix": MIXC}))
if dist_on:
    chain.append(("Distortion", {"drive": DRIVE, "threshold": D_THRESH, "mix": MIXDST}))
if drv_on:
    chain.append(("Overdrive", {"gain": G, "tone": TONE, "mix": MIXO}))
if eq_on:
    chain.append(("Equalizer", {"low_gain": LOW, "mid_gain": MID, "high_gain": HI, "mix": 1.0}))
if trem_on:
    chain.append(("Tremolo", {"rate": T_RATE, "depth": T_DEPTH, "mix": MIXT}))
if chor_on:
    chain.append(("Chorus", {"rate": C_RATE, "depth_ms": C_DEPTH, "mix": MIXC_CH}))
if d_on:
    chain.append(("Delay", {"delay_ms": DMS, "feedback": FB, "mix": MIXD}))
if r_on:
    chain.append(("Reverb", {"mix": MIXR, "pre_delay_ms": PRD, "ir_path": "assets/impulse_responses/room.wav"}))

st.markdown("---")
st.markdown(
    "**Signal Path:** 🎸 "
    + (" → ".join([f"[{n}]" for n, _ in chain]) if chain else "(dry)")
    + " → 🔊"
)

auto = st.checkbox("Auto-update visuals when knobs change", value=True, key="auto")

# Process Logic
if st.button("🚀 Process Chain", key="process_btn") or auto:
    if chain:
        y_fx = process_chain(y, sr, chain)
    else:
        y_fx = y
        
    sf.write("processed.wav", y_fx, sr)
    st.session_state["y_fx"] = y_fx

# -----------------------------
# 6. Audio Players & Visuals
# -----------------------------
# -----------------------------
# 6. Audio Players & Visuals
# -----------------------------
st.subheader("🔊 Listen & Visualize")
c1, c2 = st.columns(2)

with c1:
    st.markdown("Original")
    # FIX: Remove sample_rate=sr. The file object/path already has this info.
    # We check if data exists to avoid errors if nothing is loaded.
    source = uploaded if uploaded else "original.wav" if 'y' in locals() else None
    if source:
        st.audio(source)

with c2:
    st.markdown("Processed")
    if "y_fx" in st.session_state:
        # FIX: Remove sample_rate=sr here as well. "processed.wav" is a file path.
        st.audio("processed.wav")
    else:
        st.caption("_Process to hear result_")

# Visualization Tabs
if "y_fx" in st.session_state:
    y_fx = st.session_state["y_fx"]
    
    # [CHANGE 2] Added Tab 4 for Transfer Function
    tab1, tab2, tab3, tab4 = st.tabs(["Waveform", "Spectrogram", "Frame Explorer", "Transfer Function"])

    with tab1:
        st.pyplot(wave_compare(y, y_fx, sr))

    with tab2:
        st.pyplot(spec_compare(y, y_fx, sr))

    with tab3:
        frame_ms = st.slider("Frame length (ms)", 40, 250, 120, 10, key="frame_len")
        fig, total = frame_slider_plot(y, y_fx, sr, frame_ms=frame_ms, frame_index=0)
        idx = st.slider("Frame Index", 0, max(0, total - 1), 0, key="frame_idx")
        fig, _ = frame_slider_plot(y, y_fx, sr, frame_ms=frame_ms, frame_index=idx)
        st.pyplot(fig)

    with tab4:
        st.caption("**Input vs. Output (Phase Portrait)** - Visualizes the linearity of the effect.")
        ds = st.slider("Downsample Factor (Speed vs Detail)", 1, 50, 10, key="trans_ds")
        st.pyplot(transfer_compare(y, y_fx, downsample=ds))

# -----------------------------
# 7. Tuner
# -----------------------------
st.subheader("🎯 Tuner")
seg = y[: min(len(y), int(sr * 0.5))]

f0 = estimate_f0(seg, sr)
name, cents, freq = f0_to_note_cents(f0)
deg = max(-50, min(50, cents)) * 1.8

html_path = Path("components/tuner_gauge.html")
if html_path.exists():
    html = html_path.read_text()
    html = (
        html.replace("{{deg}}", str(deg))
        .replace("{{note}}", name)
        .replace("{{freq}}", f"{freq:.2f}" if np.isfinite(freq) else "—")
        .replace("{{cents}}", f"{int(cents):+d}")
    )
    st.components.v1.html(html, height=180)
else:
    st.info(f"Note: {name} ({freq:.1f} Hz) | {int(cents):+d} cents")