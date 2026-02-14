a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a breast tumor is malignant or benign using diagnostic features extracted from medical imaging data.

The project demonstrates a complete end-to-end machine learning workflow including:

1. Data preprocessing
2. Exploratory Data Analysis
3. Model training
4. Model evaluation using multiple metrics
5. Model comparison
6. Deployment using Streamlit

b. Dataset Description

Dataset Name: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository
Dataset Characteristics:

* Total Instances: 569
* Total Features: 30 numerical features
* Target Classes:

1. 0 → Malignant
2. 1 → Benign

* No missing values
* All features are continuous

The dataset contains computed features from digitized images of Fine Needle Aspirate (FNA) of breast masses.

c. Models Used
The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

All models were evaluated using:

1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

Model Comparison Table

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|-|-|-|-|-|-|-|
|Logistic Regression|0.973684|0.997380|0.972222|0.985915|0.979021|0.943898|
|Decision Tree|0.938596|0.932362|0.944444|0.957746|0.951049|0.868860|
|kNN|0.947368|0.981985|0.957746|0.957746|0.957746|0.887979|
|Naive Bayes|0.964912|0.997380|0.958904|0.985915|0.972222|0.925285|
|Random Forest (Ensemble)|0.964912|0.997380|0.958904|0.985915|0.972222|0.925285|
|XGBoost (Ensemble)|0.956140|0.990829|0.958333|0.971831|0.965035|0.906379|

Observations on Model Performance

|ML Model Name|Observation about model performance|
|-|-|
|Logistic Regression|Achieved the highest overall performance with excellent AUC (0.9973) and highest MCC (0.9439). Indicates strong linear separability in the dataset. Performs very stable and reliable.|
|Decision Tree|Lowest accuracy among all models. Slightly lower AUC and MCC suggest possible overfitting and sensitivity to data splits.|
|kNN|Good performance with balanced precision and recall. Sensitive to feature scaling but performs consistently after normalization.|
|Naive Bayes|Very strong AUC (0.9973). Performs well despite independence assumption. Slightly lower than Logistic Regression in MCC.|
|Random Forest (Ensemble)|High accuracy and robustness due to ensemble averaging. Performance close to Logistic Regression with strong recall and AUC.|
|XGBoost (Ensemble)|Strong performance with high AUC (0.9908). Slightly lower than Random Forest and Logistic Regression but still very powerful due to boosting mechanism.|
