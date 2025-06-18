import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

st.title("🤖 Pelatihan Model Prediksi Kelulusan")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

le = LabelEncoder()
df["Pekerjaan Sambil Kuliah"] = le.fit_transform(df["Pekerjaan Sambil Kuliah"])
df["Kategori Kehadiran"] = le.fit_transform(df["Kategori Kehadiran"])

X = df.drop("Status Kelulusan", axis=1)
y = df["Status Kelulusan"]

# Slider untuk proporsi data uji
test_size = st.slider("Pilih proporsi data uji (%)", 10, 90, 20, step=10) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Model Random Forest
model_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Model Logistic Regression sebagai perbandingan
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Menampilkan hasil
st.subheader("Akurasi Model")
st.write(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
st.write(f"Logistic Regression: {accuracy_score(y_test, y_pred_lr):.2f}")

st.subheader("Confusion Matrix - Random Forest")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Classification Report - Random Forest")
st.text(classification_report(y_test, y_pred_rf))

st.subheader("Feature Importance - Random Forest")
importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Penting": model_rf.feature_importances_
}).sort_values(by="Penting", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="Penting", y="Fitur", ax=ax)
st.pyplot(fig)
