import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.title("🔮 Prediksi Kelulusan Mahasiswa")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

st.markdown(
    """
    ✏️ Silakan masukkan data Anda untuk memprediksi peluang kelulusan berdasarkan model *Random Forest Classifier*🔍
    """
)

le_pekerjaan = LabelEncoder()
le_kehadiran = LabelEncoder()

df["Pekerjaan Sambil Kuliah"] = le_pekerjaan.fit_transform(df["Pekerjaan Sambil Kuliah"])
df["Kategori Kehadiran"] = le_kehadiran.fit_transform(df["Kategori Kehadiran"])

X = df.drop("Status Kelulusan", axis=1)
y = df["Status Kelulusan"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with st.form("form_prediksi"):
    ipk = st.slider("IPK", 2.0, 4.0, 3.0, 0.01)
    tidak_lulus = st.slider("Mata Kuliah Tidak Lulus", 0, 10, 1)
    cuti = st.slider("Jumlah Cuti Akademik", 0, 5, 1)
    pekerjaan = st.selectbox("Pekerjaan Sambil Kuliah", le_pekerjaan.classes_)
    semester = st.slider("Jumlah Semester", 6, 14, 10)
    ips_rata = st.slider("IPS Rata-rata", 2.0, 4.0, 3.0, 0.01)
    ips_akhir = st.slider("IPS Semester Akhir", 2.0, 4.0, 3.0, 0.01)
    tren = st.slider("IPS Tren", -2.0, 2.0, 0.0, 0.1)
    kehadiran = st.selectbox("Kategori Kehadiran", le_kehadiran.classes_)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame([{
        "IPK": ipk,
        "Mata Kuliah Tidak Lulus": tidak_lulus,
        "Jumlah Cuti Akademik": cuti,
        "Pekerjaan Sambil Kuliah": le_pekerjaan.transform([pekerjaan])[0],
        "Jumlah Semester": semester,
        "IPS Rata-rata": ips_rata,
        "IPS Semester Akhir": ips_akhir,
        "IPS Tren": tren,
        "Kategori Kehadiran": le_kehadiran.transform([kehadiran])[0]
    }])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][pred]

    if pred == 1:
        st.success(f"Mahasiswa diprediksi **LULUS** 🎓 (Probabilitas: {prob:.2f})")
    else:
        st.error(f"Mahasiswa diprediksi **TIDAK LULUS** ❌ (Probabilitas: {prob:.2f})")
