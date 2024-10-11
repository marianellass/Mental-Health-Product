import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


data = 'Impact_of_Remote_Work_on_Mental_Health.csv'
df = pd.read_csv(data)
print(df.head())
print("hello")


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Relevant features
features = ['Age', 'Work_Location', 'Hours_Worked_Per_Week', 'Work_Life_Balance_Rating', 
            'Number_of_Virtual_Meetings', 'Productivity_Change', 'Social_Isolation_Rating', 
            'Satisfaction_with_Remote_Work', 'Company_Support_for_Remote_Work', 
            'Physical_Activity', 'Sleep_Quality', 'Region']
target = 'Mental_Health_Condition'

# Pick columns that are categorical
categorical_columns = ['Work_Location', 'Productivity_Change', 'Satisfaction_with_Remote_Work', 
                       'Physical_Activity', 'Sleep_Quality', 'Region']

df_encoded = df.copy()
le = LabelEncoder()

# Encode categorical columns
for col in categorical_columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Encode target
df_encoded['Mental_Health_Condition'] = df_encoded['Mental_Health_Condition'].apply(
    lambda x: 1 if x == 'Burnout' or x == 'Anxiety' else 0)

# Splitting the dataset into training and test sets
X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train_scaled, y_train)

# Predict using the Logistic Regression model
y_pred = logreg_model.predict(X_test_scaled)
classification_report_output = classification_report(y_test, y_pred, target_names=['None', 'Burnout/Anxiety'])
print(classification_report_output)

# Save the trained model and the fitted scaler
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(logreg_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)