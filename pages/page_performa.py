import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("🤖 Pelatihan Model Prediksi Kelulusan (Optimasi)")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Pekerjaan Sambil Kuliah", "Kategori Kehadiran"], drop_first=True)

# Distribusi label
st.subheader("Distribusi Status Kelulusan")
st.bar_chart(df["Status Kelulusan"].value_counts())

X = df_encoded.drop("Status Kelulusan", axis=1)
y = df_encoded["Status Kelulusan"]

# Slider proporsi uji
test_size = st.slider("Pilih proporsi data uji (%)", 10, 90, 20, step=10) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Random Forest dengan class_weight
model_rf = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight="balanced", random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Logistic Regression sebagai pembanding
model_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

st.subheader("Akurasi Model")
st.write(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
st.write(f"Logistic Regression: {accuracy_score(y_test, y_pred_lr):.2f}")

st.subheader("Confusion Matrix - Random Forest")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Classification Report - Random Forest")
st.text(classification_report(y_test, y_pred_rf))

st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Penting": model_rf.feature_importances_
}).sort_values(by="Penting", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="Penting", y="Fitur", ax=ax)
st.pyplot(fig)
