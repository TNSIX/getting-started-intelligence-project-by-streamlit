import streamlit as st

st.title("Machine Learning Documentation")

# Data Preparation
st.write("<br>", unsafe_allow_html=True)
st.write("<h3>Data Preparation <span>—</span> การเตรียมข้อมูล</h3>", unsafe_allow_html=True)
st.markdown("- The data that I used in this project is a dataset of flowers that I created by using Python from code below:")
st.code("""
import pandas as pd
import numpy as np
import random

# รายการข้อมูลพื้นฐาน
flowers = ['Rose', 'Tulip', 'Sunflower', 'Orchid', 'Daisy', 'Lotus', 'Lavender', 'Marigold']
colors = {'Rose': ['Red', 'Pink', 'White'], 'Tulip': ['Yellow', 'Red', 'Purple'], 'Sunflower': ['Yellow'],
          'Orchid': ['Purple', 'White'], 'Daisy': ['White', 'Yellow'], 'Lotus': ['Pink', 'White'],
          'Lavender': ['Purple'], 'Marigold': ['Orange', 'Yellow']}
heights = {'Rose': 60, 'Tulip': 45, 'Sunflower': 150, 'Orchid': 30, 'Daisy': 25, 'Lotus': 80, 'Lavender': 40, 'Marigold': 50}
scents = ['Strong', 'Mild', 'None']
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
petal_shapes = {'Rose': 'Oval', 'Tulip': 'Elongated', 'Sunflower': 'Round', 'Orchid': 'Oval', 
                'Daisy': 'Round', 'Lotus': 'Round', 'Lavender': 'Elongated', 'Marigold': 'Star-shaped'}

# สร้างข้อมูล
data = {'Flower_Name': [], 'Color': [], 'Average_Height_cm': [], 'Scent': [], 'Bloom_Season': [], 'Petal_Shape': []}
for _ in range(3000):
    flower = random.choice(flowers)
    data['Flower_Name'].append(flower)
    data['Color'].append(random.choice(colors[flower]))
    data['Average_Height_cm'].append(heights[flower] + random.uniform(-5, 5))
    data['Scent'].append(random.choice(scents))
    data['Bloom_Season'].append(random.choice(seasons))
    data['Petal_Shape'].append(petal_shapes[flower])

# แปลงเป็น DataFrame
df = pd.DataFrame(data)

# เพิ่ม NaN แบบสุ่ม
mask = np.random.random(df.shape) < 0.07
df = df.mask(mask)

# บันทึกเป็นไฟล์ CSV
df.to_csv('flowers_petal_shape_dataset.csv', index=False)
print("Dataset ถูกบันทึกเป็น flowers_petal_shape_dataset.csv")
""")

st.divider()

# Theory of Algorithms
st.write("<h3>Theorem of Algorithms <span>—</span> ทฤษฎีของอัลกอริทึมที่พัฒนา</h3>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)

# SVM
st.markdown("""
##### 🔸Support Vector Machine (SVM)
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks, though primarily for classification. The main objective of SVM is to find the optimal hyperplane that best separates data points of different classes in a high-dimensional space.

###### Key Concepts:
1. **Hyperplane**: A decision boundary that separates different classes. In 2D, it’s a line; in 3D, it’s a plane; and in higher dimensions, it’s a hyperplane.
2. **Support Vectors**: The data points closest to the hyperplane. These points are critical in defining the position and orientation of the hyperplane.
3. **Margin**: The distance between the hyperplane and the nearest data point from either class. SVM aims to maximize this margin for better generalization.

###### How SVM Works:
- SVM seeks the hyperplane that maximizes the margin between the classes.
- For linearly separable data, it finds a hard margin. For non-linearly separable data, it uses a **soft margin** by introducing a penalty parameter (C) to allow some misclassifications.
- To handle non-linear data, SVM uses the **kernel trick**, transforming the data into a higher-dimensional space where a linear boundary can be established. Common kernels include:
    - Linear Kernel
    - Polynomial Kernel
    - Radial Basis Function (RBF) Kernel variables (for soft margin)

###### Advantages:
- Effective in high-dimensional spaces.
- Works well with both linear and non-linear data (via kernels).
- Robust to overfitting when properly tuned.

###### Disadvantages:
- Computationally intensive for large datasets.
- Sensitive to the choice of kernel and parameters (e.g., \(C\), kernel parameters).
""")
st.write("<br>", unsafe_allow_html=True)

# KNN
st.markdown("""
##### 🔸K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks, though it is more commonly applied to classification. It is a non-parametric, lazy learning algorithm that makes predictions based on the similarity (distance) between data points.

###### Key Concepts:
1. **Distance Metric**: KNN relies on a distance function (e.g., Euclidean, Manhattan) to measure the similarity between data points. The most common is Euclidean distance:
""")
st.latex("d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
st.markdown("""
2. **K Value**: The number of nearest neighbors considered for making a prediction. The choice of \(k\) affects the model's performance and decision boundary.
3. **Majority Voting (Classification)**: For classification, KNN assigns the class that is most common among the \(k\) nearest neighbors.

###### How KNN Works:
- Given a new, unseen data point, KNN calculates the distance between this point and all points in the training dataset.
- It identifies the \(k\) closest data points (neighbors) based on the chosen distance metric.
- For classification, it assigns the class with the majority vote among the \(k\) neighbors. For regression, it averages the values of the \(k\) neighbors.
- No explicit training phase is required; the algorithm stores the training data and performs computations only at prediction time.

###### Advantages:
- Simple and intuitive to understand and implement.
- No training phase, making it fast to set up.
- Naturally handles multi-class problems.

###### Disadvantages:
- Computationally expensive during prediction, especially with large datasets, as it requires calculating distances to all training points.
- Sensitive to the choice of \(k\) and the distance metric.
- Performance degrades with noisy data or irrelevant features unless preprocessed properly.
""")

st.divider()

# Model Training
st.write("<h3>Model Development <span>—</span> การพัฒนาโมเดล</h3>", unsafe_allow_html=True)
st.write("1. Import libraries")
st.code("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
""")
st.write("2. Load the dataset")
st.code("""
df = pd.read_csv('flowers_petal_shape_dataset.csv')
""")
st.write("3. Exploratory Data Analysis (EDA)")
st.code("""
# Data Formatting
print("=== Original Dataset ===")
print(df.head())
print("\\n=== Checking for Missing Values ===")
print(df.isnull().sum())
""")
st.code("""
# Drop rows with any NaN values and overwrite to df (Dataframe)
original_size = df.shape[0]
df = df.dropna()
print("\\n=== Dataset After Dropping Rows with NaN ===")
print(df.head())
print("\\n=== Checking for Missing Values Again===")
print(df.isnull().sum())
print("\\n")      
""")
st.code("""
# Count Plot for Flower_Name
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Flower_Name')
plt.title('Distribution of Flower Types')
plt.xlabel('Flower Name')
plt.ylabel('Number of Flower')
plt.xticks(rotation=45)
plt.show()

# Box Plot for Average_Height_cm by Flower_Name
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Flower_Name', y='Average_Height_cm')
plt.title('Height Distribution by Flower Type')
plt.xlabel('Flower Name')
plt.ylabel('Average Height (cm)')
plt.xticks(rotation=45)
plt.show()
""")
st.write("4. Data Preprocessing")
st.code("""
# Separate features (X) and label (y)
X = df.drop('Flower_Name', axis=1)
y = df['Flower_Name']

# Encode Categorical Features
categorical_cols = ['Color', 'Scent', 'Bloom_Season', 'Petal_Shape']
label_encoders = {}
for column in categorical_cols:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode the Label (Target Variable)
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Normalize Numerical Feature
scaler = StandardScaler()
X['Average_Height_cm'] = scaler.fit_transform(X[['Average_Height_cm']])
""")
st.code("""
# Scatter Plot after Normalize Numerical Feature
plt.figure(figsize=(10, 6))
test_df = X_test.copy()
test_df['Actual'] = le_y.inverse_transform(y_test)
test_df['Predicted_SVM'] = le_y.inverse_transform(y_pred_svm)
sns.scatterplot(data=test_df, x='Average_Height_cm', y='Actual', hue='Predicted_SVM', palette='Set2', s=100)
plt.title('After Scaled Height')
plt.xlabel('Scaled Average Height')
plt.ylabel('Flower Name')
plt.legend(title='Flower Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
""")
st.code("""
# Split Data into Training (70%) and Testing Sets (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
""")
st.write("5. Build and Train Models Using SVM and KNN")
st.code("""
# SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print('\\n=== SVM Results ===')
print(f"Accuracy: {svm_accuracy * 100:.2f}%")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=le_y.classes_))
""")
st.code("""
# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluate KNN
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print('\\n=== KNN Results ===')
print(f"Accuracy: {knn_accuracy * 100:.2f}%")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=le_y.classes_))
""")
