import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("--- Starting Diabetes Prediction Model Pipeline ---\n")

# --- 1. Data Loading ---
print("1. Loading Data...")
df = pd.read_csv("diabetes.csv")
print(f"Dataset loaded. Shape: {df.shape}\n")

# --- 2. Data Preparation ---
print("2. Preparing Data...")
# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data split into training (X_train: {X_train.shape}, y_train: {y_train.shape}) and testing (X_test: {X_test.shape}, y_test: {y_test.shape}) sets.\n")

# --- 3. Model Training (Logistic Regression) ---
print("3. Training Logistic Regression Model...")
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=200)
model.fit(X_train, y_train)
print("Model training complete!\n")

# --- 4. Model Evaluation ---
print("4. Evaluating Model Performance...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Precision: {precision:.4f}")
print(f"Model Recall: {recall:.4f}")
print(f"Model F1-Score: {f1:.4f}")

print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 5. Interactive Prediction ---
print("\n--- 5. Interactive Diabetes Prediction ---\n")
print("Please enter the values for the following features to get a prediction:")

user_input_data = {}
feature_names = X.columns # Get the column names from X

for feature in feature_names:
    while True:
        try:
            value = float(input(f"Enter value for '{feature}': "))
            user_input_data[feature] = value
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Convert user input to a DataFrame, ensuring correct column order
user_df = pd.DataFrame([user_input_data], columns=feature_names)

# Make prediction
user_prediction = model.predict(user_df)
user_prediction_proba = model.predict_proba(user_df)

print("\n--- Prediction Results ---")
if user_prediction[0] == 1:
    print("Based on the input, the model predicts that the individual is LIKELY to have diabetes.")
else:
    print("Based on the input, the model predicts that the individual is UNLIKELY to have diabetes.")

print(f"Probability of No Diabetes (Outcome = 0): {user_prediction_proba[0][0]:.4f}")
print(f"Probability of Diabetes (Outcome = 1): {user_prediction_proba[0][1]:.4f}")

print("\n--- Diabetes Prediction Model Pipeline Complete ---")
