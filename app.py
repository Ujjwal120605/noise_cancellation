import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from utils import denoise_audio_lms, denoise_audio_ml, hybrid_filter

st.title("🎧 Noise Canceller App")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    with open("noisy.wav", "wb") as f:
        f.write(uploaded_file.read())

    y, sr = librosa.load("noisy.wav", sr=None)

    st.audio("noisy.wav", format="audio/wav")

    if st.button("Denoise Audio"):
        lms_out = denoise_audio_lms(y[:10000])  # use 10k samples to avoid hang
        ml_out = denoise_audio_ml(y[:10000])
        hybrid_out = hybrid_filter(lms_out, ml_out)

        sf.write("lms.wav", lms_out, sr)
        sf.write("ml.wav", ml_out, sr)
        sf.write("hybrid.wav", hybrid_out, sr)

        st.subheader("LMS Output")
        st.audio("lms.wav", format="audio/wav")

        st.subheader("ML Output")
        st.audio("ml.wav", format="audio/wav")

        st.subheader("Hybrid Output")
        st.audio("hybrid.wav", format="audio/wav")

