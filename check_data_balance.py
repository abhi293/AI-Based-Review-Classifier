import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
train_file = os.path.join(BASE_DIR, "data/train.csv")  # Absolute path

if not os.path.exists(train_file):
    print(f"Error: {train_file} not found.")
else:
    df = pd.read_csv(train_file, header=None, names=['label', 'title', 'text'])
    print("Label Distribution:")
    print(df['label'].value_counts())

    print("\nPercentage Distribution:")
    print((df['label'].value_counts() / len(df)) * 100)
