import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Flower Classification Analysis")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload flowers_petal_shape_dataset.csv (optional)", type=['csv'])

def main():
    # Default file path (modify this to your CSV file's location)
    default_file_path = "datasets/flowers_petal_shape_dataset.csv"  # Change this to your file's path
    
    try:
        # Try loading from default path first
        df = pd.read_csv(default_file_path)
        st.sidebar.success(f"Loaded default file from: {default_file_path}")
    except FileNotFoundError:
        # If default file not found, use uploaded file or show warning
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Loaded uploaded file")
        else:
            st.warning("Default file not found. Please upload flowers_petal_shape_dataset.csv")
            return

    # Data Overview
    st.write("<h3>Date Overview</h3>", unsafe_allow_html=True)
    st.write("- ##### Original Dataset")
    st.write(df.head())
    
    st.write("- ##### Missing Values Before Cleaning")
    st.write(df.isnull().sum())
    
    # Data Preprocessing
    original_size = df.shape[0]
    df = df.dropna()
    
    st.write("- ##### After Dropping NaN Values")
    
    st.write(df.head())
    st.write(f"Rows dropped: {original_size - df.shape[0]}")
    st.write("- ##### Missing Values After Cleaning")
    st.write(df.isnull().sum())

    # Visualizations
    st.write("- ##### Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Count Plot
        fig1 = plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Flower_Name')
        plt.title('Distribution of Flower Types')
        plt.xlabel('Flower Name')
        plt.ylabel('Number of Flowers')
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    
    with col2:
        # Box Plot
        fig2 = plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Flower_Name', y='Average_Height_cm')
        plt.title('Height Distribution by Flower Type')
        plt.xlabel('Flower Name')
        plt.ylabel('Average Height (cm)')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    # Data Preparation
    X = df.drop('Flower_Name', axis=1)
    y = df['Flower_Name']

    # Encode Categorical Features
    categorical_cols = ['Color', 'Scent', 'Bloom_Season', 'Petal_Shape']
    label_encoders = {}
    for column in categorical_cols:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Encode Target
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    # Normalize Numerical Feature
    scaler = StandardScaler()
    X['Average_Height_cm'] = scaler.fit_transform(X[['Average_Height_cm']])

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # SVM
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)

    st.write("- ##### After Scaling Height")
    fig3 = plt.figure(figsize=(10, 6))
    test_df = X_test.copy()
    test_df['Actual'] = le_y.inverse_transform(y_test)
    test_df['Predicted_SVM'] = le_y.inverse_transform(y_pred_svm)
    sns.scatterplot(data=test_df, x='Average_Height_cm', y='Actual', 
                   hue='Predicted_SVM', palette='Set2', s=100)
    plt.title('After Scaled Height')
    plt.xlabel('Scaled Average Height')
    plt.ylabel('Flower Name')
    plt.legend(title='Predicted', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig3)

    # Models
    st.write("- ##### Model Results")

    col3, col4 = st.columns(2)
    with col3:
        # SVM Results
        st.write("##### SVM Results")
        st.write(f"Accuracy: {svm_accuracy * 100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_svm, target_names=le_y.classes_))
    
    with col4:
        # KNN Results
        st.write("##### KNN Results")
        st.write(f"Accuracy: {knn_accuracy * 100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_knn, target_names=le_y.classes_))

if __name__ == '__main__':
    main()