import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the trained model
model = joblib.load("model.pkl")

# Load the test data (assuming it's in the same format as train.csv)
test_file = "data/test.csv"  # Change this to the actual test data path
df_test = pd.read_csv(test_file, header=None, names=['label', 'title', 'text'])

# Handle missing values
df_test['text'] = df_test['text'].fillna("")

# Load the same TfidfVectorizer used in training
vectorizer = joblib.load("vectorizer.pkl")  # You need to save the vectorizer during training
X_test = vectorizer.transform(df_test['text'])  # Transform test data

# Get true labels
y_test = df_test['label']

# Predict
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
