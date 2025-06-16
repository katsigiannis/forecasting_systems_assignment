import pandas as pd

path = "../dataset/credit_score.csv"
dataset = pd.read_csv(path)
print(dataset.head())

print("Dataset dimensions:", dataset.shape)
print("Column names:", dataset.columns.tolist())
