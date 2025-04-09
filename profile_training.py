import sys
import os
import time
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from training.train_model import train_logistic_model as train_model
from training.preprocess import preprocess_data

# Path to your dataset
train_file = os.path.join(os.path.dirname(__file__), "data/train.csv")

# Load a small sample
sample_size = 10000  # 10K rows
df = pd.read_csv(train_file, nrows=sample_size, header=None)

# Measure preprocessing time
start_time = time.time()
X_sample, y_sample, vectorizer = preprocess_data(train_file)

preprocess_time = time.time() - start_time

# Measure training time
start_time = time.time()
train_model(X_sample, y_sample, vectorizer)  # Ensure `train_model` is a function in `train_model.py`
train_time = time.time() - start_time

# Estimate for full dataset
total_rows = 1600000
estimated_preprocess_time = (preprocess_time / sample_size) * total_rows
estimated_train_time = (train_time / sample_size) * total_rows

# Print estimated times
print(f"Preprocessing time for {sample_size} rows: {preprocess_time:.2f} sec")
print(f"Estimated time for full dataset: {estimated_preprocess_time / 60:.2f} min")
print(f"Training time for {sample_size} rows: {train_time:.2f} sec")
print(f"Estimated time for full dataset: {estimated_train_time / 60:.2f} min")
