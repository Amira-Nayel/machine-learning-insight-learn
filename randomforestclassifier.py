# -*- coding: utf-8 -*-
"""RandomForestClassifier.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-c0Tg7-BG6gWPHnD0NwaThotFW0fIckn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
import pickle

# Load the data
final_merged = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/fffffffff_(2)[1].csv')

# Drop unnecessary columns
cols_drop = ['Full Name', 'Username', 'Seat No.', 'MOT Midterm Grade', 'Time taken', 'userEmail']
df_drop = final_merged.drop(columns=cols_drop)

# Drop rows with missing 'Final SA&D' and 'attention_min'
df_Nulldrop = df_drop.dropna(subset=['Final SA&D', 'attention_min'])

# Ensure 'Final SA&D' is of float type
df_Nulldrop['Final SA&D'] = df_Nulldrop['Final SA&D'].astype(float)

# Define features to scale
features_to_scale = ['grade_and_time', 'arousal_min', 'arousal_max', 'attention_min', 'attention_max',
                     'valence_max', 'valence_min', 'volume_min', 'volume_max', 'arousal', 'attention',
                     'valence', 'volume']


# Select features and labels
X = df_Nulldrop[features_to_scale]

# Define a function to assign letter grades
def assign_letter_grade(grade):
    if grade >= 90:
        return 'A'
    elif grade >= 80:
        return 'B'
    elif grade >= 70:
        return 'C'
    elif grade >= 60:
        return 'D'
    else:
        return 'F'

# Apply the function to the target column
y = df_Nulldrop['Final SA&D'].apply(assign_letter_grade)

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Transform the training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply RandomOverSampler to balance the classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_resampled, y_resampled)

# Predict using the best estimator
y_pred = random_search.best_estimator_.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Display the classification report for more detailed metrics
print(classification_report(y_test, y_pred))

# Print the confusion matrix as numbers
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=random_search.best_estimator_.classes_, yticklabels=random_search.best_estimator_.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model using pickle
with open('final_model76.pkl', 'wb') as f:
    pickle.dump(random_search.best_estimator_, f)