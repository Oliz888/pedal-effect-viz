# 🎛️ Pedal Effect Explorer (DSP Playground)

**A "White Box" visualization tool for Discrete-Time Audio Processing.**

This project is an interactive research application built with **Streamlit** and **Python**. It is designed to demystify the mathematics behind musical audio effects. Unlike "Black Box" neural networks or physical analog circuits, this tool exposes the explicit **Discrete-Time formulas** (difference equations, transfer functions, and convolution) that drive modern digital audio workstation (DAW) plugins.

-----

## 🚀 Quick Start

### Prerequisites

  * Python 3.8+
  * `pip`

### Installation & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the application
streamlit run app.py
```

-----

## 🧪 Key Features

### 1\. The Virtual Pedalboard

An interactive chain of **8 distinct audio effects**, each representing a specific class of DSP algorithm:

  * **Dynamics:** Compressor (Piecewise functions).
  * **Non-Linearity:** Distortion (Hard Clipping) & Overdrive (Hyperbolic Tangent/Soft Clipping).
  * **Spectral:** 3-Band Equalizer (Parallel Biquad Filters).
  * **Modulation:** Tremolo (AM) & Chorus (Modulated Delay Lines).
  * **Time-Space:** Delay (Feedback Difference Equations) & Reverb (Convolution with Impulse Response).

### 2\. Multi-Modal Visualization

Verify the mathematical theory with visual proof using four distinct analysis modes:

  * **Waveform:** View the time-domain envelope and dynamics.
  * **Spectrogram:** Analyze harmonic content, saturation, and frequency filtering.
  * **Frame Explorer:** Zoom into the micro-level (100ms) to see waveshaping in action (e.g., Sine wave becoming a Square wave).
  * **Transfer Function (Phase Portrait):** A scatter plot of `Input[n]` vs `Output[n]` that instantly reveals the system's linearity (clean line), non-linearity (S-curve), or memory (scatter cloud).

### 3\. "White Box" Theory

Every pedal includes an expandable **Theory** panel that displays the exact LaTeX formula used in the code, bridging the gap between musical intuition and engineering rigor.

-----

## 📂 Project Structure

```text
├── app.py                  # Main application entry point (Streamlit UI)
├── requirements.txt        # Python dependencies (numpy, librosa, soundfile, etc.)
├── utils/
│   ├── chain.py            # DSP signal processing logic (the effect algorithms)
│   ├── visualization.py    # Plotting functions (Waveform, Spectrogram, Transfer Function)
│   └── tuner.py            # Pitch detection algorithms (YIN/Autocorrelation)
├── components/
│   ├── pedal_canvas.html   # HTML/CSS template for rendering the pedal UI
│   └── tuner_gauge.html    # HTML/CSS template for the tuner visualization
└── assets/
    └── impulse_responses/  # IR files for Convolution Reverb
```

-----

## 📚 DSP Concepts Explored

This tool helps answer the research question: *"How do simple discrete-time formulas approximate complex analog phenomena?"*

| Effect Type | Mathematical Concept | Visual Signature |
| :--- | :--- | :--- |
| **Distortion** | Memoryless Non-Linearity ($\tanh$) | **S-Curve** on Transfer Function |
| **Delay** | Difference Equation ($y[n] = x[n] + \alpha y[n-D]$) | **Cloud/Scatter** on Transfer Function |
| **Tremolo** | Amplitude Modulation (LFO) | **Envelope Shaping** on Waveform |
| **Reverb** | Convolution ($y = x * h$) | **Dense Tail** on Spectrogram |

-----

## 🛠️ Built With

  * **[Streamlit](https://streamlit.io/):** For the interactive web interface.
  * **[NumPy](https://numpy.org/):** For high-performance vector math operations.
  * **[Librosa](https://librosa.org/):** For audio analysis and STFT/Spectrogram generation.
  * **[Matplotlib](https://matplotlib.org/):** For rendering static plots and transfer functions.