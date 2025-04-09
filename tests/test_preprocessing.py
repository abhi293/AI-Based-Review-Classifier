from training.preprocess import preprocess_data

# Test preprocessing with sample data
X, y = preprocess_data('data/test.csv')

print("Shape of transformed data:", X.shape)
print("Sample labels:", y[:5])
