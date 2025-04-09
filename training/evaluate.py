from sklearn.metrics import accuracy_score
from preprocess import preprocess_data
import joblib

X_test, y_test = preprocess_data('../data/test.csv')
model = joblib.load('model.pkl')

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
