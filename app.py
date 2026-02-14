import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.title("Breast Cancer Classification Web App")
st.write("Upload test data and evaluate different classification models.")

# Model Selection Dropdown
model_name = st.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# Load Model
model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = pickle.load(open(model_paths[model_name], "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# File Upload
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    if "target" not in data.columns:
        st.error("CSV must contain 'target' column.")
        st.stop()

    X = data.drop("target", axis=1)
    y_true = data["target"]

    # Scaling
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("Precision", round(precision, 4))
    col3.metric("Recall", round(recall, 4))

    col1.metric("F1 Score", round(f1, 4))
    col2.metric("AUC Score", round(auc, 4))
    col3.metric("MCC Score", round(mcc, 4))

    # Confusion Matrix

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Classification Report    

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred)
    st.text(report)