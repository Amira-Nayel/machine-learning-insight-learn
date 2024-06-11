# -*- coding: utf-8 -*-
"""Gradient Boosting Classifier_abcd.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11cGNYkapLfbzksKkG_qOLiVsA3x-UvtK
"""

#With Just SMOTE
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_save_model(data_path, model_path):
    # Load the data
    final_merged = pd.read_csv(data_path)

    # Drop unnecessary columns
    cols_to_drop = ['Full Name', 'Username', 'Seat No.', 'MOT Midterm Grade', 'Time taken', 'userEmail', 'SA&D Midterm Grade']
    df_drop = final_merged.drop(columns=cols_to_drop)

    # Drop rows with missing 'Final SA&D' and 'attention_min'
    df_Nulldrop = df_drop.dropna(subset=['Final SA&D', 'attention_min'])

    # Ensure 'Final SA&D' is of float type using .loc to avoid SettingWithCopyWarning
    df_Nulldrop.loc[:, 'Final SA&D'] = df_Nulldrop['Final SA&D'].astype(float)

    # Define features to scale
    features_to_scale = ['grade_and_time', 'arousal_min', 'arousal_max', 'attention_min', 'attention_max',
                         'valence_max', 'valence_min', 'volume_min', 'volume_max', 'arousal', 'attention',
                         'valence', 'volume']

    # Select features and labels
    X = df_Nulldrop[features_to_scale]

    def assign_letter_grade(grade, max_grade):
        percentage = (grade / max_grade) * 100
        if percentage >= 90:
            return 'A'
        elif percentage >= 80:
            return 'B'
        elif percentage >= 70:
            return 'C'
        elif percentage >= 60:
            return 'D'
        else:
            return 'F'

    # Apply the function to the target column with the required arguments
    y = df_Nulldrop['Final SA&D'].apply(assign_letter_grade, max_grade=100)

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

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Initialize the Gradient Boosting model
    gbc_model = GradientBoostingClassifier(random_state=42)

    # Define the parameter grid for GridSearchCV
    gbc_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }

    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(gbc_model, param_grid=gbc_param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    # Predict using the best estimator
    y_pred = grid_search.best_estimator_.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Display the classification report for more detailed metrics
    all_labels = ['A', 'B', 'C', 'D', 'F']
    print(classification_report(y_test, y_pred, labels=all_labels))

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Visualize the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the trained model, scaler, and imputer using pickle
    with open(model_path, 'wb') as f:
        pickle.dump({'model': grid_search.best_estimator_, 'scaler': scaler, 'imputer': imputer}, f)

# Example usage:
train_and_save_model('/content/drive/MyDrive/fffffffff (2).csv', 'trained_model_without_pca.pkl')

# With PCA & SMOTE
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Function to train the model and save it
def train_and_save_model(data_path, model_path, n_pca_components=10):
    # Load the data
    final_merged = pd.read_csv(data_path)

    # Drop unnecessary columns
    cols_to_drop = ['Full Name', 'Username', 'Seat No.', 'MOT Midterm Grade', 'Time taken', 'userEmail', 'SA&D Midterm Grade']
    df_drop = final_merged.drop(columns=cols_to_drop)

    # Drop rows with missing 'Final SA&D' and 'attention_min'
    df_Nulldrop = df_drop.dropna(subset=['Final SA&D', 'attention_min'])

    # Ensure 'Final SA&D' is of float type using .loc to avoid SettingWithCopyWarning
    df_Nulldrop.loc[:, 'Final SA&D'] = df_Nulldrop['Final SA&D'].astype(float)

    # Define features to scale
    features_to_scale = ['grade_and_time', 'arousal_min', 'arousal_max', 'attention_min', 'attention_max',
                         'valence_max', 'valence_min', 'volume_min', 'volume_max', 'arousal', 'attention',
                         'valence', 'volume']


    # Select features and labels
    X = df_Nulldrop[features_to_scale]

    # Function to assign letter grade
    def assign_letter_grade(grade, max_grade=100):
        percentage = (grade / max_grade) * 100
        if percentage >= 90:
            return 'A'
        elif percentage >= 80:
            return 'B'
        elif percentage >= 70:
            return 'C'
        elif percentage >= 60:
            return 'D'
        else:
            return 'F'

    # Apply the function to the target column with max_grade
    y = df_Nulldrop['Final SA&D'].apply(assign_letter_grade, max_grade=100)

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

    # Apply PCA
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_pca, y_train)

    # Initialize the Gradient Boosting model
    gbc_model = GradientBoostingClassifier(random_state=42)

    # Define the parameter grid for GridSearchCV
    gbc_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }

    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(gbc_model, param_grid=gbc_param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    # Predict using the best estimator
    y_pred = grid_search.best_estimator_.predict(X_test_pca)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Display the classification report for more detailed metrics
    all_labels = ['A', 'B', 'C', 'D', 'F']
    print(classification_report(y_test, y_pred, labels=all_labels))

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Visualize the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the trained model, scaler, imputer, and PCA components using pickle
    with open(model_path, 'wb') as f:
        pickle.dump({'model': grid_search.best_estimator_, 'scaler': scaler, 'imputer': imputer, 'pca': pca}, f)

# Example usage:
train_and_save_model('/content/drive/MyDrive/fffffffff (2).csv', 'trained_model2.pkl', n_pca_components=10)