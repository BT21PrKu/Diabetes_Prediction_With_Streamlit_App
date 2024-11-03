import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Load and display data
@st.cache
def load_data():
    df = pd.read_csv("diabetes_dataset.csv")
    return df

df = load_data()
st.write("### Diabetes Dataset", df.head())

if st.sidebar.checkbox("Show dataset info"):
    st.write("Data Shape:", df.shape)
    st.write("Data Types:", df.dtypes)
    st.write("Summary Statistics:", df.describe())

    st.write("## Outcome Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax)
    st.pyplot(fig)

# Data cleaning - replace 0s with mean/median for specific columns
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

# Split the data into features and target
X = df.drop("Outcome", axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model selection and prediction section
st.sidebar.write("## Model Selection")
model_choice = st.sidebar.selectbox("Choose a model:", ["Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])

# Pre-trained models with best hyperparameters
svm_model = SVC(kernel='linear', C=0.01, gamma='scale')
knn_model = KNeighborsClassifier(n_neighbors=20, p=2, weights='uniform')
rf_model = RandomForestClassifier(n_estimators=1800, max_features='sqrt')

# Model evaluation
if model_choice == "Support Vector Machine":
    model = svm_model
elif model_choice == "K-Nearest Neighbors":
    model = knn_model
elif model_choice == "Random Forest":
    model = rf_model

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("## Model Performance")
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.write("F1 Score:", f1_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))

# User input for prediction
st.write("## Predict Diabetes")
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Creating new data for prediction
new_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

if st.button("Predict"):
    prediction = model.predict(new_data)
    outcome = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.write(f"### Prediction: {outcome}")
