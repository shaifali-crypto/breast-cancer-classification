import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer ML Classification",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #4B8BBE;
        }
        .sub-text {
            text-align: center;
            font-size: 18px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧬 Breast Cancer Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Machine Learning Model Comparison & Evaluation</p>', unsafe_allow_html=True)
st.write("---")

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
st.sidebar.header("⚙ Model Selection")

model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model_name = st.sidebar.selectbox(
    "Choose a Classification Model",
    options=list(model_paths.keys())
)

# -----------------------------
# Load Model and Scaler Properly
# -----------------------------
@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

@st.cache_resource
def load_scaler():
    return pickle.load(open("model/scaler.pkl", "rb"))

model = load_model(model_paths[model_name])
scaler = load_scaler()

# -----------------------------
# File Upload Section
# -----------------------------
st.subheader("📂 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV File (must contain 30 features + target column)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    with st.expander("🔍 Preview Uploaded Data"):
        st.dataframe(data.head())

    # Check if target column exists
    if "target" not in data.columns:
        st.error("CSV must contain 'target' column.")
        st.stop()

    X = data.drop("target", axis=1)
    y_true = data["target"]

    # -----------------------------
    # Ensure Correct Feature Alignment
    # -----------------------------
    expected_columns = scaler.feature_names_in_

    missing_cols = set(expected_columns) - set(X.columns)
    if missing_cols:
        st.error(f"Missing columns in uploaded CSV: {missing_cols}")
        st.stop()

    # Reorder columns correctly
    X = X[expected_columns]

    # Scale Features
    X_scaled = scaler.transform(X)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    st.write("---")
    st.subheader(f"📊 Evaluation Metrics — {model_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
    col2.metric("Precision", round(precision_score(y_true, y_pred), 4))
    col3.metric("Recall", round(recall_score(y_true, y_pred), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y_true, y_pred), 4))
    col5.metric("AUC Score", round(roc_auc_score(y_true, y_prob), 4))
    col6.metric("MCC Score", round(matthews_corrcoef(y_true, y_pred), 4))

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.write("---")
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    st.pyplot(fig)

    # -----------------------------
    # Classification Report
    # -----------------------------
    st.write("---")
    st.subheader("📄 Classification Report")

    report = classification_report(y_true, y_pred)
    st.code(report)
