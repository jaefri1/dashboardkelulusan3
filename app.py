import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

st.set_page_config(page_title="Dashboard Kelulusan Mahasiswa", layout="wide", page_icon="🎓")

st.title("🎓 Dashboard Kelulusan Mahasiswa")
st.markdown("### Selamat datang! 👋")

st.markdown(
    """
Kami dari **Kelompok 13** mempersembahkan sebuah dashboard interaktif yang membahas  
**Analisis Kelulusan Mahasiswa Berbasis Data Mining**📊🎯

Dashboard ini dirancang untuk memberikan wawasan mengenai karakteristik mahasiswa, mengevaluasi performa model prediksi, serta memperkirakan kelulusan berdasarkan data historis.

---

🔍 **Gunakan menu di sebelah kiri untuk menjelajahi fitur-fitur berikut:**
- 📊 **Eksplorasi Data**: Visualisasi dan penjelajahan data kelulusan
- ⚙️ **Performa Model**: Menampilkan evaluasi dari model klasifikasi yang digunakan
- 🧮 **Prediksi Kelulusan**: Formulir untuk memprediksi status kelulusan mahasiswa

---

📂 **Sumber Data:**  
[Dataset Kelulusan Mahasiswa - Kaggle](https://www.kaggle.com/datasets/afitoindrapermana/dataset-kelulusan-mahasiswa)
"""
)
st.dataframe(df)
