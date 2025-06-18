import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Eksplorasi Data")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

st.subheader("Statistik Deskriptif")
st.write(df.describe())

st.subheader("Visualisasi Distribusi Status Kelulusan")
fig, ax = plt.subplots()
sns.countplot(x="Status Kelulusan", data=df, ax=ax)
st.pyplot(fig)

st.subheader("📉 Distribusi Fitur Numerik Berdasarkan Status Kelulusan")

    fitur_numerik = ['IPK', 'IPS Rata-rata', 'IPS Semester Akhir', 
                     'IPS Tren', 'Jumlah Semester', 'Mata Kuliah Tidak Lulus']

    for fitur in fitur_numerik:
        st.markdown(f"### {fitur}")
        fig, ax = plt.subplots()
        sns.boxplot(x='Status Kelulusan', y=fitur, data=df, palette='pastel')
        ax.set_xticklabels(['Tidak Lulus', 'Lulus'])
        ax.set_title(f"Distribusi {fitur} berdasarkan Status Kelulusan")
        st.pyplot(fig)

