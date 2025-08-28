import streamlit as st
from melo.api import TTS
import nltk
nltk.download('popular')

st.title("Advanced MeloTTS Multi-Language TTS Demo")
st.write("Enter Text, choose language, speaker, and fine-tune pitch & speed, then generate & download your audio!")

#Text input with validation
text_input = st.text_area("Input Text", "Streamlit and open-source TTS are awesome!")
if len(text_input.strip()) == 0:
    st.warning("Please enter some text to generate Audio.")

# ✅ Only keep supported languages from cleaner.py
available_languages = [
    ('EN', 'English (Default)'),
    ('FR', 'French'),
    ('ES', 'Spanish'),
    ('ZH', 'Chinese'),
    ('KR', 'Korean')
]
language_codes = [code for code, label in available_languages]
language_labels = [label for code, label in available_languages]
language_idx = st.selectbox("Choose Language/Accent", range(len(language_codes)), format_func=lambda i: language_labels[i])
language = language_codes[language_idx]

speed = st.slider("Select Speed", 0.5, 2.0, 1.0)

device = 'cpu' # use cuda:0 if GPU is available

#instantiate TTS to get Speakers for the selected language with caching to improve UI responsiveness
@st.cache_resource
def get_tts(lang):
    return TTS(device=device, language=lang)
tts = get_tts(language)

speaker_ids = tts.hps.data.spk2id
speakers = list(speaker_ids.keys())
speaker = st.selectbox("Choose Speaker", speakers)

#optional: Emotion selection if your model supports it
emotions = ["Neutral", "Happy", "Sad", "Angry", "Calm"]
emotion = st.selectbox("Select Emotion", emotions)

if st.button("Generate Audio") and len(text_input.strip()) > 0:
    output_path = "output.wav"
    try:
        tts.tts_to_file(text_input, speaker_ids[speaker], output_path, speed=speed)
        
        with open(output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.download_button("Download Audio", data=audio_bytes, file_name="output.wav", mime="audio/wav")

        st.success("Audio generated successfully!")
    except Exception as e:
        st.error(f"Error generating audio: {e}")

if st.button("Reset"):
    st.experimental_rerun()
