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

    st.subheader("Visualisasi Korelasi")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.subheader("Distribusi IPK")
    fig, ax = plt.subplots()
    sns.histplot(df['IPK'], kde=True, ax=ax)
    st.pyplot(fig)
