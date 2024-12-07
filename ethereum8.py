import streamlit as st
import pandas as pd
import numpy as np  # Import NumPy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("transaction_dataset.csv")  
    return data

# Function to train models and return predictions
def train_models(X_train, y_train, model_choice):
    if model_choice == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_choice == 'Random Forest':
        model = RandomForestClassifier()
    elif model_choice == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_choice == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    return model

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(importances)])
        st.pyplot(plt)

# Streamlit Application
st.title("Ethereum Fraud Detection")

# Load data
data = load_data()

# Show dataset overview
if st.checkbox('Show dataset'):
    st.write(data.head())

# Select features for training
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect("Select Features", data.columns[:-1])  # Exclude 'FLAG' (fraud label)

# Select model
model_choice = st.sidebar.selectbox("Select Model", ('Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost'))

# Initialize report variable
report = None

# Split data into train and test
if features:
    X = data[features]
    y = data['FLAG']  # Assuming 'FLAG' is the target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = train_models(X_train, y_train, model_choice)

    # Make predictions
    predictions = model.predict(X_test)

    # Show classification report
    st.subheader(f"{model_choice} Classification Report")
    report = classification_report(y_test, predictions, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Download button for classification report
    report_df = pd.DataFrame(report).transpose()
    st.download_button(
        label="Download Classification Report",
        data=report_df.to_csv().encode('utf-8'),
        file_name='classification_report.csv',
        mime='text/csv'
    )

    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc_score = auc(recall, precision)
    st.write(f"Area Under Precision-Recall Curve: {auc_score:.2f}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_choice} Precision-Recall Curve')
    st.pyplot(plt)

    # Feature Importance
    st.subheader("Feature Importance")
    plot_feature_importance(model, features)

    # Interactive Data Visualization
    st.subheader("Data Distribution")
    if features:  # Check if features list is not empty
        for feature in features:
            if pd.api.types.is_numeric_dtype(data[feature]):  # Ensure the feature is numeric
                plt.figure(figsize=(10, 6))
                sns.histplot(data[feature], kde=True, label=feature, alpha=0.5)
                plt.legend()
                st.pyplot(plt)
            else:
                st.write(f"The feature '{feature}' is not numeric and cannot be plotted.")
    else:
        st.write("Please select features to visualize their distribution.")

# User Input for Fraud Prediction
st.sidebar.header("User  Input for Fraud Detection")
input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(f"Input {feature}", value=0.0)

if st.sidebar.button("Predict Fraud"):
    user_input_df = pd.DataFrame([input_data])
    prediction = model.predict(user_input_df)
    st.write(f"Predicted Fraud: {prediction[0]}")