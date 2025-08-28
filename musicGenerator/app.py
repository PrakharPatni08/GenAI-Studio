import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import io

# Load model once
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return processor, model

processor, model = load_model()

st.title("Advanced AI Music Generator with MusicGen")

# Predefined music genres and instruments for quick prompts
styles = [
    "uplifting cinematic ambient", "jazzy saxophone", "upbeat pop", "relaxing piano",
    "electronic dance", "rock guitar", "orchestral strings", "hip hop beat"
]

style_choice = st.selectbox("Choose Style/Instrument", styles)
custom_prompt = st.text_area("Or enter a custom prompt", style_choice)

duration = st.slider("Duration (seconds)", 5, 30, 10)
tempo = st.slider("Approximate Tempo (BPM)", 60, 180, 120)
volume = st.slider("Volume Multiplier", 0.1, 3.0, 1.0)
generate_btn = st.button("Generate Music")

def plot_waveform(audio_data):
    fig, ax = plt.subplots()
    times = np.linspace(0, len(audio_data) / 16000, len(audio_data))
    ax.plot(times, audio_data)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title("Waveform")
    st.pyplot(fig)

if generate_btn:
    with st.spinner("Generating music, please wait..."):
        # Generate music from text prompt
        inputs = processor(
            text=[custom_prompt],
            padding=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            audio_outputs = model.generate(**inputs, max_new_tokens=duration * 50)

        audio_np = audio_outputs[0].cpu().numpy().T
        # Apply volume control (simple amplification)
        audio_np = np.clip(audio_np * volume, -1, 1)

        # Save to WAV
        output_file = "advanced_generated_music.wav"
        sf.write(output_file, audio_np, 16000)

        # Show waveform
        plot_waveform(audio_np)

        # Play and download
        with open(output_file, "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format='audio/wav')
            st.download_button(
                label="Download Music",
                data=audio_bytes,
                file_name="advanced_generated_music.wav",
                mime="audio/wav"
            )
    st.success("Music generated successfully!")
