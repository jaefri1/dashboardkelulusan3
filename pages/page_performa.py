import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("🤖 Pelatihan Model Prediksi Kelulusan")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

le = LabelEncoder()
df["Pekerjaan Sambil Kuliah"] = le.fit_transform(df["Pekerjaan Sambil Kuliah"])
df["Kategori Kehadiran"] = le.fit_transform(df["Kategori Kehadiran"])

# Slider untuk menentukan rasio data pelatihan
train_size = st.slider("Pilih proporsi data pelatihan", min_value=0.1, max_value=0.9, value=0.8, step=0.1)

X = df.drop("Status Kelulusan", axis=1)
y = df["Status Kelulusan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Penting": model.feature_importances_
}).sort_values(by="Penting", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="Penting", y="Fitur", ax=ax)
st.pyplot(fig)

# Simpan model dan train_size di session_state untuk digunakan di halaman prediksi
st.session_state.model = model
st.session_state.train_size = train_size
