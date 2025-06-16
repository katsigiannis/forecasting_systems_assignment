import pandas as pd

# Load dataset from disk
CREDIT_SCORE_DATASET_PATH = "../dataset/credit_score.csv"
dataset = pd.read_csv(CREDIT_SCORE_DATASET_PATH)
print("Dataset dimensions:", dataset.shape)
print("Column names:", dataset.columns.tolist())
