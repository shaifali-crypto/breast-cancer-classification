import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report

# Page Config
st.set_page_config(
    page_title="Breast Cancer ML Classification",
    page_icon="🧬",
    layout="wide"
)

# Custom Styling
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
        .metric-box {
            padding: 10px;
            border-radius: 10px;
            background-color: #f5f5f5;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<p class="main-title">🧬 Breast Cancer Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Machine Learning Model Comparison & Evaluation</p>', unsafe_allow_html=True)
st.write("---")

# Sidebar - Model Selection
st.sidebar.header("⚙ Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# Model Paths
model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# Load model and scaler
model = pickle.load(open(model_paths[model_name], "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# File Upload Section
st.subheader("📂 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV File with Features + target column", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    with st.expander("Preview Uploaded Data"):
        st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("CSV must contain 'target' column.")
        st.stop()

    X = data.drop("target", axis=1)
    y_true = data["target"]

    # Ensure correct column alignment
    expected_columns = scaler.feature_names_in_
    missing_cols = set(expected_columns) - set(X.columns)

    if missing_cols:
        st.error(f"Missing columns in uploaded CSV: {missing_cols}")
        st.stop()

    X = X[expected_columns]
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics Section
    st.write("---")
    st.subheader("📊 Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
    col2.metric("Precision", round(precision_score(y_true, y_pred), 4))
    col3.metric("Recall", round(recall_score(y_true, y_pred), 4))

    col4, col5, col6 = st.columns(3)

    col4.metric("F1 Score", round(f1_score(y_true, y_pred), 4))
    col5.metric("AUC Score", round(roc_auc_score(y_true, y_prob), 4))
    col6.metric("MCC Score", round(matthews_corrcoef(y_true, y_pred), 4))

    # Confusion Matrix
    st.write("---")
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    st.pyplot(fig)

    # Classification Report
    st.write("---")
    st.subheader("📄 Classification Report")

    report = classification_report(y_true, y_pred)
    st.code(report)
