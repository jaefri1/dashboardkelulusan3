import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

st.set_page_config(page_title="Dashboard Kelulusan Mahasiswa", layout="wide", page_icon="🎓")

st.title("🎓 Dashboard Kelulusan Mahasiswa")
st.write("Gunakan sidebar untuk menjelajahi halaman eksplorasi, pelatihan model, dan prediksi.")
st.dataframe(df)