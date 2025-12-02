import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from utils.chain import process_chain
from utils.visualization import wave_compare, spec_compare, frame_slider_plot
from utils.tuner import estimate_f0, f0_to_note_cents

st.set_page_config(page_title="Audio FX Explorer – v5 Playground", layout="wide")
st.title("🎛️ Audio FX Explorer – Educational + Playground (v5)")

# -----------------------------
# 1. Input
# -----------------------------
uploaded = st.sidebar.file_uploader("Upload WAV/MP3", type=["wav", "mp3"], key="upl")
use_demo = st.sidebar.checkbox("Use demo sine (440 Hz)", value=not uploaded, key="demo")

if use_demo:
    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
else:
    y, sr = librosa.load(uploaded, sr=44100, mono=True)

# -----------------------------
# 2. Pedalboard visuals
# -----------------------------
st.subheader("Pedalboard")

def render_pedal(title, bg, led_on, config):
    html = Path("components/pedal_canvas.html").read_text()
    html = (
        html.replace("{{title}}", title)
        .replace("{{bg}}", bg)
        .replace("{{led}}", "on" if led_on else "")
        .replace("{{config}}", str(config).replace("'", '"'))
    )
    st.components.v1.html(html, height=380)


cols = st.columns(4, gap="large")

# -------- Compressor --------
with cols[0]:
    st.markdown("**Compressor**")
    comp_on = st.checkbox("On", value=False, key="comp_on")
    T = st.slider("Threshold", 0.05, 0.9, 0.4, key="comp_T")
    R = st.slider("Ratio", 1.0, 20.0, 4.0, key="comp_R")
    MIXC = st.slider("Mix", 0.0, 1.0, 1.0, key="comp_mix")

    # NOTE: values set to "" so nothing is drawn inside the knobs → no overlap
    render_pedal(
    "COMP",
    "#ff4d4d",  # 🔥 bright pedal red
    comp_on,
    {
        "time": min((R - 1) / 19, 1.0),
        "fb": T,
        "mix": MIXC,
        "labels": {"time": "RATIO", "fb": "THRESH", "mix": "MIX"},
        "values": {"time": "", "fb": "", "mix": ""},  # keep text empty (no overlap)
    },
)


    # Clean numeric readout BELOW the pedal
    st.markdown(
        f"""
        <div style="text-align:center; font-size:13px; line-height:1.2;">
        R = {R:.1f} &nbsp;&nbsp;|&nbsp;&nbsp; T = {T:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; Mix = {MIXC:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📘 Formula & Theory – Compressor"):
        st.latex(
            r"y = \begin{cases}x, & |x|<T\\ \operatorname{sign}(x)\left(T+\frac{|x|-T}{R}\right), & |x|\ge T\end{cases}"
        )
        st.markdown(
            f"""**T = {T:.2f}** – amplitude threshold.  
**R = {R:.1f}** – higher means stronger compression.  
**Mix = {MIXC:.2f}** – dry/wet blend."""
        )

# -------- Overdrive --------
with cols[1]:
    st.markdown("**Overdrive**")
    drv_on = st.checkbox("On", value=False, key="drv_on")
    G = st.slider("Gain (G)", 1.0, 12.0, 3.0, key="drv_G")
    TONE = st.slider("Tone", 0.0, 1.0, 0.2, key="drv_tone")
    MIXO = st.slider("Mix", 0.0, 1.0, 1.0, key="drv_mix")

    render_pedal(
        "DRIVE",
        "#2aa6ff",
        drv_on,
        {
            "time": (G - 1) / 11,
            "fb": TONE,
            "mix": MIXO,
            "labels": {"time": "GAIN", "fb": "TONE", "mix": "MIX"},
            "values": {"time": "", "fb": "", "mix": ""},  # no text inside knobs
        },
    )

    st.markdown(
        f"""
        <div style="text-align:center; font-size:13px; line-height:1.2;">
        G = {G:.1f} &nbsp;&nbsp;|&nbsp;&nbsp; Tone = {TONE:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; Mix = {MIXO:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📘 Formula & Theory – Overdrive"):
        st.latex(r"y = (1 - \text{mix})x + \text{mix}\tanh(Gx)")
        st.markdown(
            f"""**G = {G:.2f}** – pre-gain into the waveshaper.  
**Tone = {TONE:.2f}** – brightness emphasis.  
**Mix = {MIXO:.2f}** – dry/wet blend."""
        )

# -------- Delay --------
with cols[2]:
    st.markdown("**Delay**")
    d_on = st.checkbox("On", value=True, key="d_on")
    DMS = st.slider("Time (ms)", 50, 800, 300, key="d_ms")
    FB = st.slider("Feedback (α)", 0.0, 0.95, 0.4, key="d_fb")
    MIXD = st.slider("Mix", 0.0, 1.0, 0.3, key="d_mix")

    render_pedal(
        "DELAY",
        "#16c2b0",
        d_on,
        {
            "time": (DMS - 50) / 750,
            "fb": FB,
            "mix": MIXD,
            "labels": {"time": "TIME", "fb": "FEED", "mix": "MIX"},
            "values": {"time": "", "fb": "", "mix": ""},  # no overlap
        },
    )

    st.markdown(
        f"""
        <div style="text-align:center; font-size:13px; line-height:1.2;">
        Time = {DMS} ms &nbsp;&nbsp;|&nbsp;&nbsp; Feedback = {FB:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; Mix = {MIXD:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📘 Formula & Theory – Delay"):
        st.latex(r"y[n] = (1 - \text{mix})x[n] + \text{mix}(x[n] + \alpha y[n-D])")
        st.markdown(
            f"""**α = {FB:.2f}** – echo persistence.  
**D = {DMS} ms** – delay spacing.  
**Mix = {MIXD:.2f}** – wet/dry balance."""
        )

# -------- Reverb --------
with cols[3]:
    st.markdown("**Reverb**")
    r_on = st.checkbox("On", value=True, key="r_on")

    PRD = st.slider("Pre-delay (ms)", 0, 200, 0, key="r_pre")
    ROOM = st.slider("Room Size", 0.5, 2.0, 1.0, 0.05, key="r_room")   # ← NEW SLIDER
    MIXR = st.slider("Mix", 0.0, 1.0, 0.3, key="r_mix")

    render_pedal(
        "REVERB",
        "#6aa5ff",
        r_on,
        {
            "time": PRD / 200 if 200 > 0 else 0.0,
            "fb": ROOM / 2.0,   # maps to knob visually
            "mix": MIXR,
            "labels": {"time": "PRE", "fb": "SIZE", "mix": "MIX"},
            "values": {"time": "", "fb": "", "mix": ""},
        },
    )

    st.markdown(
        f"""
        <div style="text-align:center; font-size:13px; line-height:1.2;">
        Pre-delay = {PRD} ms &nbsp;&nbsp;|&nbsp;&nbsp;
        Size = {ROOM:.2f}× &nbsp;&nbsp;|&nbsp;&nbsp;
        Mix = {MIXR:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📘 Formula & Theory – Reverb"):
        st.latex(r"y = (1 - \text{mix})x + \text{mix}(x * h_{\text{room}}^{(\text{scaled})})")
        st.markdown(
            f"""**Room Size = {ROOM:.2f}×** – stretches or compresses the impulse response.  
**Pre-delay = {PRD} ms** – gap before reflections.  
**Mix = {MIXR:.2f}** – wet level.  
**h_room** – the room IR after time scaling."""
        )


# -----------------------------
# 3. Chain + processing
# -----------------------------
chain = []
if comp_on:
    chain.append(("Compressor", {"threshold": T, "ratio": R, "makeup": 1.0, "mix": MIXC}))
if drv_on:
    chain.append(("Overdrive", {"gain": G, "tone": TONE, "mix": MIXO}))
if d_on:
    chain.append(("Delay", {"delay_ms": DMS, "feedback": FB, "mix": MIXD}))
if r_on:
    chain.append(
        (
            "Reverb",
            {
                "mix": MIXR,
                "pre_delay_ms": PRD,
                "ir_path": "assets/impulse_responses/room.wav",
            },
        )
    )

st.markdown(
    "**Signal Path:** 🎸 "
    + (" → ".join([f"[{n}]" for n, _ in chain]) if chain else "(dry)")
    + " → 🔊"
)

auto = st.checkbox("Auto-update visuals when knobs change", value=True, key="auto")

if st.button("🚀 Process Chain", key="process_btn") or auto:
    y_fx = process_chain(y, sr, chain) if chain else y
    sf.write("processed.wav", y_fx, sr)
    st.session_state["y_fx"] = y_fx

# -----------------------------
# 4. Players
# -----------------------------
st.subheader("🔊 Listen")
c1, c2 = st.columns(2)

with c1:
    sf.write("original.wav", y, sr)
    st.audio("original.wav")

with c2:
    if "y_fx" in st.session_state:
        st.audio("processed.wav")
    else:
        st.caption("_Process the chain to preview_")

# -----------------------------
# 5. Visual tabs
# -----------------------------
if "y_fx" in st.session_state:
    y_fx = st.session_state["y_fx"]
    tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "Frame Explorer"])

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

# -----------------------------
# 6. Tuner
# -----------------------------
st.subheader("🎯 Tuner")
seg = y[: min(len(y), int(sr * 0.5))]

f0 = estimate_f0(seg, sr)
name, cents, freq = f0_to_note_cents(f0)
deg = max(-50, min(50, cents)) * 1.8

html = Path("components/tuner_gauge.html").read_text()
html = (
    html.replace("{{deg}}", str(deg))
    .replace("{{note}}", name)
    .replace("{{freq}}", f"{freq:.2f}" if np.isfinite(freq) else "—")
    .replace("{{cents}}", f"{int(cents):+d}")
)
st.components.v1.html(html, height=180)
