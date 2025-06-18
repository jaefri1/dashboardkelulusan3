import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.title("🤖 Pelatihan Model Prediksi Kelulusan")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

# Pra-pemrosesan
label_cols = ["Pekerjaan Sambil Kuliah", "Kategori Kehadiran"]
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Pisahkan fitur dan target
y = df["Status Kelulusan"]
X = df.drop("Status Kelulusan", axis=1)

# One-hot encoding jika perlu
X = pd.get_dummies(X, drop_first=True)

# Skala fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tangani data imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Slider untuk proporsi data uji
test_size = st.slider("Pilih proporsi data uji (%)", 10, 90, 20, step=10) / 100
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, stratify=y_res, random_state=42)

# Model 1: Random Forest
model_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Model 2: Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Model 3: XGBoost
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# Evaluasi akurasi
st.subheader("Akurasi Model")
st.write(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
st.write(f"Logistic Regression: {accuracy_score(y_test, y_pred_lr):.2f}")
st.write(f"XGBoost: {accuracy_score(y_test, y_pred_xgb):.2f}")

# Confusion Matrix untuk model terbaik
st.subheader("Confusion Matrix - XGBoost")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report - XGBoost")
st.text(classification_report(y_test, y_pred_xgb))

# Feature Importance
st.subheader("Feature Importance - XGBoost")
importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Penting": model_xgb.feature_importances_
}).sort_values(by="Penting", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="Penting", y="Fitur", ax=ax)
st.pyplot(fig)
