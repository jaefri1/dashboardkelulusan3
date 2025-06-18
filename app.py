import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

st.set_page_config(page_title="Dashboard Kelulusan Mahasiswa", layout="wide", page_icon="🎓")

st.title("🎓 Dashboard Kelulusan Mahasiswa")
st.write("Gunakan sidebar untuk berpindah ke halaman-halaman berikut")
st.write("halaman 1 page eksplorasi yang berisi dataset dan karakteristiknya atau EDA (exploratory data analysis).") 
st.write("halaman 2 page performa yang berisi berisi hasil pelatihan model.")
st.write("halaman 3 page prediksi yang berisi formulir untuk melakukan prediksi dan hasilnya.")
st.dataframe(df)
