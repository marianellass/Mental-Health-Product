"""This file creates a hybrid model combining SVM and Random Forest for predicting Depression and saves the necessary components for deployment with Streamlit."""

import pandas as pd
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import class_weight
from imblearn.combine import SMOTEENN

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load data
data = 'Impact_of_Remote_Work_on_Mental_Health.csv'
df = pd.read_csv(data)
print(df.head())
print("Data loaded successfully.")

# Fill missing values in Mental_Health_Condition with "None"
df['Mental_Health_Condition'] = df['Mental_Health_Condition'].fillna('None')

# Print counts of each condition to verify
condition_counts = df['Mental_Health_Condition'].value_counts()
print("Condition counts:\n", condition_counts)

# Relevant features
features = [
    'Years_of_Experience', 'Industry', 'Work_Location', 'Hours_Worked_Per_Week',
    'Work_Life_Balance_Rating', 'Physical_Activity',
    'Sleep_Quality', 'Stress_Level'
]
target = 'Mental_Health_Condition'

# Map stress levels to numeric values using map instead of replace to avoid FutureWarning
stress_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['Stress_Level'] = df['Stress_Level'].map(stress_mapping)
print("Unique Stress Levels after mapping:", df['Stress_Level'].unique())

# Categorical columns to encode
categorical_columns = ['Industry', 'Work_Location', 'Physical_Activity', 'Sleep_Quality']

# Initialize a dictionary to hold LabelEncoders for each categorical column
encoders = {}

# Encode categorical features using separate LabelEncoders
df_encoded = df.copy()
for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le  # Save the encoder for each column

# Encode target: 1 for 'Depression', 0 for 'None'
df_encoded['Mental_Health_Condition'] = df_encoded['Mental_Health_Condition'].apply(
    lambda x: 1 if x == 'Depression' else 0
)

# Splitting the dataset into training and test sets with stratification
X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data split into training and testing sets.")

# Handle class imbalance by applying SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
print("After SMOTEENN, class distribution:\n", pd.Series(y_train_resampled).value_counts())

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed.")

# Initialize SVM with class weights and probability estimates
svm_model = SVC(
    random_state=42,
    class_weight='balanced',
    probability=True  # Enable probability estimates for soft voting
)

# Initialize Random Forest with class weights
rf_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# Create a Voting Classifier combining SVM and Random Forest
voting_clf = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('rf', rf_model)
    ],
    voting='soft'  # 'soft' voting uses predicted probabilities
)

# Train the Voting Classifier on resampled training data
voting_clf.fit(X_train_scaled, y_train_resampled)
print("Voting Classifier training completed.")

# Predict using the Voting Classifier
y_pred = voting_clf.predict(X_test_scaled)

# Generate classification report
classification_report_output = classification_report(
    y_test, y_pred, target_names=['None', 'Depression'], zero_division=0
)
print("Voting Classifier Classification Report:")
print(classification_report_output)

# Calculate and print Accuracy and F1-Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {accuracy:.2f}")
print(f"Voting Classifier F1-Score: {f1:.2f}")

# Save the trained Voting Classifier model
with open('voting_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(voting_clf, model_file)
print("Voting Classifier model saved as 'voting_classifier_model.pkl'.")

# Save the scaler for deployment
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("Scaler saved successfully as 'scaler.pkl'.")

# Save the encoders for deployment
with open('encoders.pkl', 'wb') as enc_file:
    pickle.dump(encoders, enc_file)
print("LabelEncoders saved successfully as 'encoders.pkl'.")

