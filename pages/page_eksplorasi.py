import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Eksplorasi Data Kelulusan Mahasiswa")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

st.markdown(
"""
Halaman ini menyajikan analisis deskriptif dan visualisasi untuk memahami pola dan hubungan antar fitur dalam dataset kelulusan mahasiswa. Analisis ini membantu dalam tahap awal memahami struktur data sebelum membangun model prediksi.
"""
)

st.markdown(
"""
---
## 📌 Statistik Deskriptif

Berikut adalah ringkasan statistik untuk fitur numerik dalam dataset:
"""
)
st.dataframe(df.describe(), use_container_width=True)

st.markdown(
    """
---
## 📈 Distribusi Status Kelulusan

Visualisasi ini menampilkan jumlah mahasiswa yang **lulus** dan **tidak lulus** berdasarkan label di dataset.
"""
)
fig1, ax1 = plt.subplots()
sns.set_style("whitegrid")
sns.countplot(x="Status Kelulusan", data=df, palette="pastel", ax=ax1)
ax1.set_title("Distribusi Status Kelulusan", fontsize=14)
ax1.set_xlabel("Status Kelulusan")
ax1.set_ylabel("Jumlah Mahasiswa")
st.pyplot(fig1)

st.markdown(
    """
---
## 🔍 Korelasi Antar Fitur

Peta panas berikut menggambarkan tingkat korelasi antar fitur numerik dalam dataset.
Gradasi warna dari biru ke merah menunjukkan kekuatan dan arah hubungan antar variabel, di mana warna merah menunjukkan korelasi positif yang kuat, sedangkan biru menunjukkan korelasi negatif.
"""
)
fig2, ax2 = plt.subplots(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2, linewidths=0.5)
st.pyplot(fig2)
