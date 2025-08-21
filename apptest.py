import io
import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import streamlit as st

# ---- optional deps (safe fallbacks if missing)
try:
    import soundfile as sf  # for clean WAV writing
    HAVE_SF = True
except Exception:
    HAVE_SF = False

try:
    import cv2
    from fer import FER
    HAVE_EMOTION = True
except Exception:
    HAVE_EMOTION = False

APP_TITLE = "🎵 Generative Music & Mood Visualizer"
SAMPLE_RATE = 44_100

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------------------------
# Mood presets & simple NLP
# ---------------------------
# ---------------------------
# Mood presets & simple NLP
# ---------------------------
MOOD_PRESET = {
    # --- Original Moods ---
    "happy":   {"tempo": 120, "scale": "major", "wave": "saw",      "reverb": 0.2,  "energy": 0.8},
    "calm":    {"tempo": 75,  "scale": "major", "wave": "sine",     "reverb": 0.35, "energy": 0.3},
    "sad":     {"tempo": 70,  "scale": "minor", "wave": "triangle", "reverb": 0.25, "energy": 0.4},
    "lofi":    {"tempo": 85,  "scale": "minor", "wave": "sine",     "reverb": 0.4,  "energy": 0.5},
    "dark":    {"tempo": 100, "scale": "minor", "wave": "square",   "reverb": 0.15, "energy": 0.9},
    "epic":    {"tempo": 140, "scale": "minor", "wave": "saw",      "reverb": 0.1,  "energy": 1.0},
    "focus":   {"tempo": 100, "scale": "major", "wave": "sine",     "reverb": 0.1,  "energy": 0.5},
    "romance": {"tempo": 95,  "scale": "major", "wave": "triangle", "reverb": 0.3,  "energy": 0.6},
    
    # --- New Moods ---
    "energetic": {"tempo": 135, "scale": "major", "wave": "saw",    "reverb": 0.15, "energy": 0.95},
    "ambient": {"tempo": 65,  "scale": "major", "wave": "sine",     "reverb": 0.6,  "energy": 0.2},
    "mysterious": {"tempo": 90, "scale": "minor", "wave": "triangle", "reverb": 0.3, "energy": 0.6},
    "aggressive": {"tempo": 150, "scale": "minor", "wave": "square", "reverb": 0.1, "energy": 1.0},
    "peaceful": {"tempo": 60,  "scale": "major", "wave": "sine",     "reverb": 0.5, "energy": 0.1},
}

def tokenize_mood(text: str) -> List[str]:
    return [w.strip().lower() for w in text.replace(",", " ").split() if w.strip()]

def mood_to_params(text: str) -> Dict[str, Any]:
    tokens = tokenize_mood(text)
    if not tokens:
        return MOOD_PRESET["happy"].copy()

    scores = {k: 0 for k in MOOD_PRESET}
    synonyms = {
        "happy":   ["joy", "uplifting", "cheerful", "bright", "sunny"],
        "calm":    ["chill", "soft", "ambient", "relax", "meditative"],
        "sad":     ["blue", "melancholy", "down", "emo", "cry"],
        "lofi":    ["lo-fi", "study", "chilled", "dusty", "vinyl"],
        "dark":    ["ominous", "gritty", "tense", "industrial", "noir"],
        "epic":    ["trailer", "heroic", "cinematic", "power", "massive"],
        "focus":   ["work", "deep", "neutral", "concentration", "steady"],
        "romance": ["love", "warm", "tender", "date", "heart"],
          # --- Keywords for New Moods ---
        "energetic": ["dance", "party", "upbeat", "fast", "rave", "techno", "active", "driving"],
        "ambient": ["space", "drone", "atmospheric", "expansive", "nebula", "texture", "background"],
        "mysterious": ["noir", "detective", "suspense", "intrigue", "shadowy", "unsolved", "curious"],
        "aggressive": ["industrial", "angry", "intense", "power", "rage", "harsh", "metal", "attack"],
        "peaceful": ["meditate", "serene", "tranquil", "zen", "stillness", "healing", "yoga", "reverie"],
    }

    for t in tokens:
        if t in scores:
            scores[t] += 2
        for mood, syns in synonyms.items():
            if t in syns:
                scores[mood] += 1

    # Find the best matching mood, or default to the first token if it's a direct match
    top_mood = max(scores, key=scores.get)
    if scores[top_mood] == 0 and tokens[0] in MOOD_PRESET:
        top_mood = tokens[0]
    
    params = MOOD_PRESET[top_mood].copy()


    # small token-driven tweaks
    if any(t in tokens for t in ["fast", "upbeat", "energetic", "dance"]):
        params["tempo"] = min(180, params["tempo"] + 25)
        params["energy"] = min(1.0, params["energy"] + 0.2)
    if any(t in tokens for t in ["slow", "sleep", "soft", "ambient"]):
        params["tempo"] = max(60, params["tempo"] - 20)
        params["energy"] = max(0.1, params["energy"] - 0.2)
    if "minor" in tokens or "dark" in tokens:
        params["scale"] = "minor"
    if "major" in tokens or "bright" in tokens:
        params["scale"] = "major"
    return params

def get_scale_notes(scale_type: str) -> List[int]:
    if scale_type == "minor":
        return [0, 3, 5, 7, 10, 12]  # Minor Pentatonic
    return [0, 2, 4, 7, 9, 12]      # Major Pentatonic

# ---------------------------
# Advanced Synth Building Blocks & FILTERS
# ---------------------------

def simple_lowpass_filter(signal: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """A simple IIR low-pass filter to tame harsh frequencies."""
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]
    return filtered

def simple_highpass_filter(signal: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """A simple IIR high-pass filter to remove low-end rumble."""
    filtered = np.zeros_like(signal)
    last_in = 0
    last_out = 0
    for i in range(len(signal)):
        filtered[i] = alpha * (last_out + signal[i] - last_in)
        last_in = signal[i]
        last_out = filtered[i]
    return filtered

def osc(wave: str, freq_hz: float, length_samples: int) -> np.ndarray:
    t = np.arange(length_samples) / SAMPLE_RATE
    output = np.zeros(length_samples, dtype=np.float32)
    
    # Generate raw waveform
    if wave == "saw":
        T = 1.0 / max(freq_hz, 1e-6)
        output = 2.0 * ((t % T) / T) - 1.0
        # **FIX**: Apply a strong filter to remove harsh "buzzy" harmonics
        output = simple_lowpass_filter(output, alpha=0.3)
    elif wave == "square":
        output = np.sign(np.sin(2 * np.pi * freq_hz * t))
        # **FIX**: Also filter the square wave to make it less harsh
        output = simple_lowpass_filter(output, alpha=0.3)
    elif wave == "triangle":
        T = 1.0 / max(freq_hz, 1e-6)
        saw = 2.0 * ((t % T) / T) - 1.0
        output = 2.0 * np.abs(saw) - 1.0
    else: # "sine" is the default
        output = np.sin(2 * np.pi * freq_hz * t)
        
    return output.astype(np.float32)

def apply_envelope(signal: np.ndarray, attack=0.01, release=0.1) -> np.ndarray:
    n = len(signal)
    a = int(attack * SAMPLE_RATE)
    r = int(release * SAMPLE_RATE)
    env = np.ones(n, dtype=np.float32)
    
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    # **FIX**: Ensure release doesn't go out of bounds
    if r > 0 and r < n:
        env[-r:] = np.linspace(1, 0, r)
    elif r >= n:
        env[:] = np.linspace(1, 0, n)
        
    return signal * env

def simple_reverb(signal: np.ndarray, mix=0.2, delay_ms=120, feedback=0.5) -> np.ndarray:
    """Slightly improved reverb with feedback."""
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    if delay_samples <= 0 or delay_samples >= len(signal):
        return signal
        
    wet_signal = np.zeros_like(signal)
    for i in range(delay_samples, len(signal)):
        wet_signal[i] = signal[i - delay_samples] + wet_signal[i - delay_samples] * feedback
        
    peak = np.abs(wet_signal).max()
    if peak > 1.0:
        wet_signal /= peak
        
    return (1 - mix) * signal + mix * wet_signal

# ---------------------------
# Advanced Procedural Track Generator
# ---------------------------
# ---------------------------
# Advanced Procedural Track Generator (CORRECTED)
# ---------------------------
def generate_track(params: Dict[str, Any], seconds: int = 20, root_freq: float = 220.0) -> np.ndarray:
    # --- Parameters ---
    tempo, wave, scale_type, energy, reverb_amount = \
        params["tempo"], params["wave"], params["scale"], params["energy"], params["reverb"]

    # --- Timing ---
    beats_per_sec = tempo / 60.0
    steps_per_beat = 2  # 8th notes
    steps = int(seconds * beats_per_sec * steps_per_beat)
    step_len = int(SAMPLE_RATE / (beats_per_sec * steps_per_beat))
    total_len = steps * step_len
    
    # --- Musical ---
    notes = get_scale_notes(scale_type)
    rng = np.random.default_rng(seed=int(energy * 1000) + int(tempo))

    # --- Instrument Tracks ---
    melody = np.zeros(total_len, dtype=np.float32)
    bass   = np.zeros(total_len, dtype=np.float32)
    hats   = np.zeros(total_len, dtype=np.float32)
    pad    = np.zeros(total_len, dtype=np.float32)

    # --- Part 1: Bass on downbeats ---
    bass_freq = root_freq / 2.0
    for i in range(0, steps, 4):
        start = i * step_len
        tone_len = step_len * 2
        tone = osc("sine", bass_freq, tone_len)
        enveloped_tone = apply_envelope(tone * 0.7, attack=0.01, release=0.3)
        
        # **FIX**: Safely write the note to the track, preventing overflow
        end_pos = min(start + tone_len, total_len)
        actual_len = end_pos - start
        bass[start:end_pos] += enveloped_tone[:actual_len]

    # --- Part 2: Hi-Hats ---
    for i in range(steps):
        if i % 2 == 0:
            start = i * step_len
            noise = rng.standard_normal(step_len).astype(np.float32) * 0.1
            filtered_noise = simple_highpass_filter(noise, alpha=0.95)
            # This part is safe as it never exceeds step_len, but we keep the logic consistent
            end_pos = min(start + step_len, total_len)
            actual_len = end_pos - start
            hats[start:end_pos] += apply_envelope(filtered_noise, attack=0.001, release=0.03)[:actual_len]

    # --- Part 3: Atmospheric Pad (NEW) ---
    pad_chord = [notes[0], notes[2], notes[4]]
    pad_wave = "triangle" if wave != "sine" else "sine"
    measure_len_steps = steps_per_beat * 4
    for i in range(0, steps, measure_len_steps):
        start = i * step_len
        tone_len = measure_len_steps * step_len
        
        for degree in pad_chord:
            freq = root_freq * (2 ** (degree / 12.0))
            tone = osc(pad_wave, freq, tone_len)
            enveloped_tone = apply_envelope(tone * 0.2, attack=0.5, release=0.5)

            # **FIX**: Safely write the pad chord note, preventing overflow
            end_pos = min(start + tone_len, total_len)
            actual_len = end_pos - start
            pad[start:end_pos] += enveloped_tone[:actual_len]

    # --- Part 4: Smarter Melody ---
    current_note_index = rng.integers(0, len(notes))
    for i in range(steps):
        if rng.random() < (0.5 + 0.4 * energy):
            start = i * step_len
            step_change = rng.choice([-2, -1, -1, 0, 1, 1, 2])
            current_note_index = np.clip(current_note_index + step_change, 0, len(notes) - 1)
            
            semis = notes[current_note_index]
            if rng.random() < 0.15 + 0.2 * energy:
                semis += 12
            
            freq = root_freq * (2 ** (semis / 12.0))
            velocity = 0.6 + rng.random() * 0.4
            
            tone_len = int(step_len * 1.5) # This was the source of the error
            tone = osc(wave, float(freq), tone_len)
            enveloped_tone = apply_envelope(tone * velocity, attack=0.005, release=0.15)
            
            # **FIX**: The crucial fix for the melody track that caused the error
            end_pos = min(start + tone_len, total_len)
            actual_len = end_pos - start
            melody[start:end_pos] += enveloped_tone[:actual_len]
    
    # --- Final Mixing ---
    mix = (melody * 0.5) + (bass * 0.9) + (hats * 0.25) + (pad * 0.7)
    mix = simple_reverb(mix, mix=reverb_amount, delay_ms=150, feedback=0.4)
    
    mix = simple_lowpass_filter(mix, alpha=0.9)
    mix = np.clip(mix, -1.0, 1.0)

    peak = np.abs(mix).max()
    if peak > 1e-5:
        mix = (mix / peak) * 0.9
        
    return mix.astype(np.float32)

# ---------------------------------------------
# WAV helpers & UI (NO CHANGES NEEDED BELOW)
# ---------------------------------------------
def wav_bytes_from_np(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    if HAVE_SF:
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        return buf.getvalue()
    # fallback: tiny PCM16 writer
    pcm = (audio * np.iinfo(np.int16).max).astype("<i2").tobytes()
    byte_rate = sample_rate * 2
    block_align = 2
    data_size = len(pcm)
    riff_size = 36 + data_size
    header = (
        b"RIFF" + (riff_size).to_bytes(4, "little") + b"WAVE" +
        b"fmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little") +
        (1).to_bytes(2, "little") + (sample_rate).to_bytes(4, "little") +
        (byte_rate).to_bytes(4, "little") + (block_align).to_bytes(2, "little") +
        (16).to_bytes(2, "little") + b"data" + (data_size).to_bytes(4, "little")
    )
    return header + pcm

def make_visualizer_html(audio_b64: str, title: str = "Generated Track") -> str:
    # <audio> + WebAudio analyser -> canvas bars, synced to playback
    return f"""
    <div style="display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;">
      <audio id="player" controls src="data:audio/wav;base64,{audio_b64}"></audio>
      <strong style="font-size:.95rem">{title}</strong>
    </div>
    <canvas id="viz" width="900" height="200"
            style="width:100%;height:200px;background:#0b1020;border-radius:12px;margin-top:8px;"></canvas>
    <script>
      const audio = document.getElementById('player');
      const canvas = document.getElementById('viz');
      const ctx = canvas.getContext('2d');
      let started = false, audioCtx, analyser, source, dataArray;

      function setup(){{
        if(started) return;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        dataArray = new Uint8Array(analyser.frequencyBinCount);
        source = audioCtx.createMediaElementSource(audio);
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
        started = true;
      }}

      audio.addEventListener('play', setup);

      function draw(){{
        requestAnimationFrame(draw);
        if(!started) return;
        analyser.getByteFrequencyData(dataArray);
        const w=canvas.width, h=canvas.height;
        ctx.clearRect(0,0,w,h);
        const bars = dataArray.length;
        const barW = (w / bars) * 1.4;
        for(let i=0;i<bars;i++){{
          const v = dataArray[i] / 255.0;
          const bh = v*h;
          const x = i*barW;
          ctx.fillStyle = `hsl(${{200 + i*0.7}}, 80%, ${{40 + v*40}}%)`;
          ctx.fillRect(x, h-bh, barW*0.8, bh);
        }}
        // little timing orb
        const t = (audio.currentTime % 4) / 4;
        ctx.beginPath(); ctx.arc(30 + t*(w-60), 28, 10, 0, Math.PI*2); ctx.fillStyle='#fff8'; ctx.fill();
      }}
      draw();
    </script>
    """

if "playlist" not in st.session_state:
    st.session_state["playlist"] = []

st.title(APP_TITLE)
st.caption("Type a mood → generate music → watch the visualizer. Optional: webcam emotion bias.")

col_left, col_right = st.columns([1.2, 0.8], gap="large")

with col_left:
    with st.form("generator"):
        mood_text = st.text_input("Mood / keywords", placeholder="e.g., calm lofi study, happy upbeat, dark epic")
        dur = st.slider("Duration (seconds)", 10, 60, 20, 5)
        root = st.selectbox("Root note", ["A","B","C","D","E","F","G"], index=0)
        root_freqs = {"A":220.0,"B":246.94,"C":261.63,"D":293.66,"E":329.63,"F":349.23,"G":392.00}

        adv = st.expander("Advanced options")
        detect_webcam = adv.checkbox("Use webcam emotion (optional)", value=False,
                                     help="Needs opencv-python + fer. If unavailable, this is ignored.")
        submitted = st.form_submit_button("🎶 Generate")

    if submitted:
        webcam_emotion = None
        if detect_webcam and HAVE_EMOTION:
            try:
                detector = FER(mtcnn=True)
                cap = cv2.VideoCapture(0)
                ok, frame = cap.read()
                cap.release()
                if ok:
                    result = detector.top_emotion(frame)
                    if result and result[0]:
                        webcam_emotion = result[0]
            except Exception:
                webcam_emotion = None

        mood_in = (mood_text or "").strip()
        if not mood_in and webcam_emotion:
            mood_in = webcam_emotion
        elif webcam_emotion:
            mood_in = f"{mood_in} {webcam_emotion}"

        with st.spinner(f"Generating '{mood_in}' track..."):
            params = mood_to_params(mood_in)
            audio = generate_track(params, seconds=dur, root_freq=root_freqs[root])
            wav_bytes = wav_bytes_from_np(audio, SAMPLE_RATE)
            b64 = base64.b64encode(wav_bytes).decode()
            title = f"{mood_in or 'untitled'} • {datetime.now().strftime('%H:%M:%S')}"

            st.markdown(make_visualizer_html(b64, title), unsafe_allow_html=True)

            item = {
                "title": title,
                "mood": mood_in or "n/a",
                "params": params,
                "duration": int(dur),
                "created_at": datetime.now().isoformat(),
                "wav_b64": b64,
            }
            st.session_state["playlist"].insert(0, item) # Insert at beginning

            st.download_button("⬇️ Download WAV", data=wav_bytes, file_name=f"{title.replace(' ', '_')}.wav", mime="audio/wav")
            with st.expander("Generation parameters"):
                st.json(params)

with col_right:
    st.subheader("📜 Playlist History")
    if not st.session_state["playlist"]:
        st.info("No tracks yet. Generate something!")
    else:
        # Display playlist items without reversing the list
        for i, it in enumerate(st.session_state["playlist"]):
            with st.container(border=True):
                st.write(f"**{it['title']}**")
                st.caption(f"Mood tags: `{it['mood']}` • {int(it['duration'])}s")
                st.audio(base64.b64decode(it["wav_b64"]), format="audio/wav")
                st.download_button(
                    "Download",
                    data=base64.b64decode(it["wav_b64"]),
                    file_name=f"track_{i}_{it['mood']}.wav",
                    mime="audio/wav",
                    key=f"dl_{i}"
                )

    st.divider()
    st.subheader("📤 Export Playlist")
    if st.session_state["playlist"]:
        export = json.dumps(st.session_state["playlist"], indent=2)
        st.download_button("Download playlist (.json)", data=export, file_name="playlist.json", mime="application/json")

st.markdown("---")
with st.expander("ℹ️ Advanced: Diffusion-based text→music (Riffusion)"):
    st.write("""
**How to integrate:**
1) Install a compatible `torch` build (preferably with CUDA) and the riffusion pipeline from its repo/wheel.
2) Generate a spectrogram image from your prompt, then convert it back to audio (Griffin–Lim).
3) Replace the `generate_track(...)` call with the Riffusion pipeline output.

Pseudo-snippet:
```python
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.riffusion_pipeline import RiffusionPipeline

pipe = RiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
image = pipe(prompt=mood_text).images[0]
converter = SpectrogramImageConverter()
segment = converter.audio_from_spectrogram_image(image)
audio = segment.samples
""") 