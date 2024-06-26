# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14trLC7EmP3g4Df1L448FAZkgl3l1Y5QJ
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the data
#final_merged = pd.read_excel('/content/fffffffff.xlsx')
final_merged = pd.read_csv('data.csv')
# Drop specified columns
cols_drop = ['Full Name', 'Username', 'Seat No.', 'MOT Midterm Grade', 'Time taken', 'userEmail']
df_drop = final_merged.drop(columns=cols_drop)

# Drop rows where 'SA&D Midterm Grade' is null
df_Nulldrop = df_drop.dropna(subset=['SA&D Midterm Grade'])
df_Nulldrop = df_Nulldrop.dropna(subset=['attention_min'])

# Convert 'SA&D Midterm Grade' to float and multiply by 4.0
df_Nulldrop['SA&D Midterm Grade'] = df_Nulldrop['SA&D Midterm Grade'].astype(float)
df_Nulldrop['SA&D Midterm Grade'] = df_Nulldrop['SA&D Midterm Grade'] * 4.0

# Define features to scale
features_to_scale = [ 'grade_and_time','arousal_min', 'arousal_max', 'attention_min', 'attention_max',
                     'valence_max', 'valence_min', 'volume_min', 'volume_max', 'arousal', 'attention',
                     'valence', 'volume']

# Create feature matrix X and target vector y
X = df_Nulldrop[features_to_scale]
y = df_Nulldrop['SA&D Midterm Grade'].apply(lambda x: 1 if x >= 65 else 0)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.4, random_state=42, stratify=y)

# Oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data
scaler.fit(X_train_resampled)

# Transform the training and test data
X_train_scaled = scaler.transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test data
y_pred_logreg = logistic_model.predict(X_test_scaled)

# Evaluate the Logistic Regression model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", accuracy_logreg)

# Confusion Matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# Print classification report for Logistic Regression
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logreg))

# Initialize and train the SVM model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test data
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for SVM')
plt.show()

# Print classification report for SVM
print("Classification Report for SVM:")
print(classification_report(y_test, y_pred_svm))