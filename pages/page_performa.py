import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("🤖 Pelatihan Model Prediksi Kelulusan")

@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")
    return df

df = load_data()

st.markdown(
    """
    Pada halaman ini akan ditunjukkan tingkat akurasi dari model yang telah dipilih, 
    di mana aplikasi ini menggunakan model "Random Forest Classifier* 📈🧮
    """
)

# Tampilkan preview data
st.write("### Contoh data:")
st.dataframe(df.head())

# Encoding fitur kategorikal
categorical_cols = df.select_dtypes(include='object').columns.drop("Status Kelulusan")
df_encoded = df.copy()
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Simpan encoder kalau nanti dibutuhkan

# Encoding target
le_target = LabelEncoder()
df_encoded["Status Kelulusan"] = le_target.fit_transform(df["Status Kelulusan"])

# Tentukan fitur dan target
X = df_encoded.drop("Status Kelulusan", axis=1)
y = df_encoded["Status Kelulusan"]

# Pilihan proporsi data uji
test_size = st.slider("Pilih proporsi data uji (%)", 10, 50, 20, step=10) / 100

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Model 1: Random Forest
model_rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Model 2: Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Tampilkan akurasi
st.subheader("Akurasi Model")
st.write(f"🎯 Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
st.write(f"📈 Logistic Regression: {accuracy_score(y_test, y_pred_lr):.2f}")

# Confusion Matrix - Random Forest
st.subheader("Confusion Matrix - Random Forest")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Classification Report - Random Forest
st.subheader("Classification Report - Random Forest")
st.text(classification_report(y_test, y_pred_rf))

# Feature Importance
st.subheader("Feature Importance - Random Forest")
importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Penting": model_rf.feature_importances_
}).sort_values(by="Penting", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=importance_df, x="Penting", y="Fitur", ax=ax)
st.pyplot(fig)
