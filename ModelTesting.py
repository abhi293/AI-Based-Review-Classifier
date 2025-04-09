import joblib
import pandas as pd
from training.preprocess import preprocess_data
from sklearn.metrics import classification_report, accuracy_score

# Load test data
X_test, y_test = preprocess_data('data/test.csv')

# Load trained model
model = joblib.load('model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
