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

# Encoding
le = LabelEncoder()
df["Pekerjaan Sambil Kuliah"] = le.fit_transform(df["Pekerjaan Sambil Kuliah"])
df["Kategori Kehadiran"] = le.fit_transform(df["Kategori Kehadiran"])

# Target dan fitur
X = df.drop("Status Kelulusan", axis=1)
y = df["Status Kelulusan"]

# Tampilkan distribusi label
total_lulus = y.value_counts()
st.subheader("Distribusi Label Status Kelulusan")
st.bar_chart(total_lulus)

# Slider untuk memilih proporsi data latih
test_size = st.slider("Pilih persentase data untuk pengujian", min_value=0.1, max_value=0.9, step=0.1, value=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Akurasi
st.write(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Visualisasi y_test vs y_pred
st.subheader("Perbandingan Nilai Sebenarnya dan Prediksi")
compare_df = pd.DataFrame({"Aktual": y_test.values, "Prediksi": y_pred})
st.dataframe(compare_df.reset_index(drop=True))

# Classification Report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Feature Importance
st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Penting": model.feature_importances_
}).sort_values(by="Penting", ascending=False)

fig2, ax2 = plt.subplots()
sns.barplot(data=importance_df, x="Penting", y="Fitur", ax=ax2)
st.pyplot(fig2)
