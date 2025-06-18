import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif

st.title("🚀 Pelatihan Model Prediksi Kelulusan (Optimized)")

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_kelulusan_mahasiswa.csv")

df = load_data()

# Analisis data
st.subheader("Analisis Data")
st.write(f"Jumlah sampel: {len(df)}")
st.write(f"Jumlah fitur: {len(df.columns) - 1}")
st.write("Distribusi kelas:")
st.bar_chart(df["Status Kelulusan"].value_counts())

# Pisahkan fitur dan target
y = df["Status Kelulusan"]
X = df.drop("Status Kelulusan", axis=1)
X = pd.get_dummies(X)  # Otomatis mengubah fitur kategorik ke numerik

# UI controls
st.sidebar.header("Pengaturan Model")
test_size = st.sidebar.slider("Pilih proporsi data uji (%)", 10, 90, 20, step=10) / 100
feature_selection = st.sidebar.checkbox("Aktifkan Seleksi Fitur", value=True)
use_smote = st.sidebar.checkbox("Gunakan SMOTE untuk Imbalanced Data", value=True)
model_choice = st.sidebar.selectbox("Pilih Model", ["Random Forest", "Gradient Boosting", "Logistic Regression"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
if feature_selection:
    k = st.sidebar.slider("Jumlah fitur terbaik yang dipilih", 5, min(50, len(X.columns)), min(20, len(X.columns)))
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
    X_test_scaled = selector.transform(X_test_scaled)
    
    # Dapatkan nama fitur yang dipilih
    selected_features = X.columns[selector.get_support()]
    st.sidebar.write(f"Fitur terpilih ({k}):")
    st.sidebar.write(list(selected_features))
else:
    selected_features = X.columns

# Handle class imbalance with SMOTE
if use_smote:
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    st.sidebar.success(f"SMOTE diterapkan. Distribusi kelas setelah: {pd.Series(y_train).value_counts().to_dict()}")

# Model training
if model_choice == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier(random_state=42)
elif model_choice == "Gradient Boosting":
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    }
    model = GradientBoostingClassifier(random_state=42)
else:  # Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    model = LogisticRegression(max_iter=1000, random_state=42)

# Hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None

# Tampilkan hasil
st.subheader("Hasil Pelatihan")
st.write(f"Model terbaik: **{model_choice}**")
st.write(f"Parameter terbaik: {grid_search.best_params_}")
st.write(f"Akurasi Cross-Validation: {grid_search.best_score_:.4f}")
st.write(f"Akurasi pada Data Uji: **{accuracy_score(y_test, y_pred):.4f}**")

if y_proba is not None:
    st.write(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title(f"Confusion Matrix - {model_choice}")
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.highlight_max(axis=0))

# Feature Importance
if hasattr(best_model, "feature_importances_"):
    st.subheader("Feature Importance")
    
    importance_df = pd.DataFrame({
        "Fitur": selected_features,
        "Penting": best_model.feature_importances_
    }).sort_values(by="Penting", ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Penting", y="Fitur", ax=ax)
    ax.set_title("15 Fitur Paling Penting")
    st.pyplot(fig)

# ROC Curve (jika model mendukung probabilitas)
if y_proba is not None:
    from sklearn.metrics import RocCurveDisplay
    
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)
    ax.set_title('ROC Curve')
    st.pyplot(fig)

# Simpan model
if st.button("Simpan Model Terbaik"):
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    if feature_selection:
        joblib.dump(selector, 'feature_selector.pkl')
    st.success("Model dan preprocessing objects disimpan!")
