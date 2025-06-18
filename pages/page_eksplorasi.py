import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

    st.title("📊 Eksplorasi Data Kelulusan Mahasiswa")
    st.markdown("""
    Halaman ini menyajikan eksplorasi data awal (EDA) dari dataset kelulusan mahasiswa untuk memahami distribusi dan pola antar fitur.
    """)

    st.subheader("🔍 Dataset")
    st.dataframe(df)

    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    st.markdown("---")
    st.subheader("📊 Visualisasi Data")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Status Kelulusan")
        fig, ax = plt.subplots()
        sns.countplot(x='Status Kelulusan', data=df, palette='Set2')
        ax.set_xticklabels(['Tidak Lulus', 'Lulus'])
        ax.set_title("Jumlah Mahasiswa per Status Kelulusan")
        st.pyplot(fig)

    with col2:
        st.markdown("### Korelasi antar Fitur Numerik")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        st.pyplot(fig)

    st.markdown("---")
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

    st.markdown("---")
    st.subheader("👥 Distribusi Fitur Kategorikal")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pekerjaan Sambil Kuliah")
        fig, ax = plt.subplots()
        sns.countplot(x='Pekerjaan Sambil Kuliah', data=df, hue='Status Kelulusan', palette='Set1')
        ax.set_title("Pekerjaan Sambil Kuliah vs Status Kelulusan")
        st.pyplot(fig)

    with col2:
        st.markdown("### Kategori Kehadiran")
        fig, ax = plt.subplots()
        sns.countplot(x='Kategori Kehadiran', data=df, hue='Status Kelulusan', palette='Set3')
        ax.set_title("Kategori Kehadiran vs Status Kelulusan")
        st.pyplot(fig)
