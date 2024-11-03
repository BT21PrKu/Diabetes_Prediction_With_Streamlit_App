# Diabetes_Prediction_With_Streamlit_App

## Table of Contents
1. [Introduction](#introduction)
2. [Project Objective](#project-objective)
3. [Dataset Overview](#dataset-overview)
4. [Project Workflow](#project-workflow)
5. [Machine Learning Models Used](#machine-learning-models-used)
6. [Streamlit App](#streamlit-app)
7. [Technologies Used](#technologies-used)
8. [Results and Performance Metrics](#results-and-performance-metrics)

---

## Introduction

Diabetes is a chronic illness that impairs the body's ability to manage blood glucose levels. This project leverages machine learning to predict whether an individual has diabetes based on medical variables. Accurate predictions can support early diagnosis and management of diabetes, leading to better patient outcomes.

## Project Objective

The primary goal of this project is to classify individuals as diabetic or non-diabetic using various machine learning algorithms and to deploy a user-friendly app for real-time predictions.

## Dataset Overview

The dataset contains medical features and an outcome variable indicating whether an individual has diabetes. Key attributes are as follows:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body Mass Index ((weight in kg)/(height in m)^2)
- **DiabetesPedigreeFunction**: A score indicating the likelihood of diabetes based on family history
- **Age**: Age in years
- **Outcome**: Binary outcome (1 = diabetic, 0 = non-diabetic)

## Project Workflow

1. **Data Import and Exploration**: Import libraries, load data, and understand dataset structure.
2. **Data Cleaning**: Handle missing values by replacing 0s with mean or median values for key features.
3. **Data Visualization**: Generate visualizations to understand distributions, relationships, and identify any outliers.
4. **Feature Selection**: Identify important features for classification.
5. **Handling Outliers**: Address any detected outliers to ensure robust model performance.
6. **Data Splitting**: Split data into training and testing sets.
7. **Model Building and Tuning**:
   - **KNN**: K-Nearest Neighbors
   - **Naive Bayes**
   - **SVM**: Support Vector Machine
   - **Decision Tree**
   - **Random Forest**
   - **Logistic Regression**
   - Hyperparameter tuning using GridSearchCV

## Machine Learning Models Used

Each model was fine-tuned using GridSearchCV to achieve the best hyperparameters, and metrics such as F1 score, precision, and recall were used to evaluate performance. The models used include:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Logistic Regression**
- **Naive Bayes**
- **Decision Tree**

Each model was evaluated based on its performance in predicting diabetes and then integrated into the final Streamlit app.

## Streamlit App

A Streamlit app was created to provide an interactive interface for diabetes prediction. The app allows users to input personal medical information and receive a real-time prediction of their diabetes status. Key features of the app include:

- **Model Selection**: Users can choose between the top-performing models: SVM, KNN, and Random Forest.
- **Real-Time Prediction**: After entering details, users can click "Predict" to see whether they are predicted to be diabetic or non-diabetic.
- **Visualization**: The app includes basic dataset information, summary statistics, and a distribution of diabetic vs. non-diabetic cases.

### App Screen-Shots
![image](https://github.com/user-attachments/assets/d7b3f295-31a8-4210-b9d4-56ac1b807be8)
![image](https://github.com/user-attachments/assets/e6784d41-6b44-4f68-8a45-7be572cc9caf)
![image](https://github.com/user-attachments/assets/9d1dbd51-cc87-4c81-9e76-8305eaeb82c1)

## Technologies Used

- **Python** (for data preprocessing, model building, and analysis)
- **Pandas** and **NumPy** (for data manipulation)
- **Seaborn** and **Matplotlib** (for data visualization)
- **Scikit-Learn** (for machine learning models and metrics)
- **Streamlit** (for app deployment and user interaction)

## Results and Performance Metrics

Each model was evaluated based on accuracy, F1 score, precision, and recall. The final app uses the best hyperparameters found for SVM, KNN, and Random Forest, allowing the user to select the desired model for making predictions.

- **Classification Report**: Displays precision, recall, and F1 score for each model.
- **Confusion Matrix**: Visualizes true positives, false positives, true negatives, and false negatives for each model.
