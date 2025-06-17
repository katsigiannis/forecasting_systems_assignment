import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset from disk
CREDIT_SCORE_DATASET_PATH = "../dataset/credit_score.csv"
df = pd.read_csv(CREDIT_SCORE_DATASET_PATH)
print("Dataset dimensions:", df.shape)
print("Column names:", df.columns.tolist())

X = df.drop(columns=['DEFAULT', 'CUST_ID'])
y = df['DEFAULT']

# Display information about missing values in X
print("\nMissing values in X:")
print(X.isnull().sum())
print("\nTotal missing values:", X.isnull().sum().sum())

# Display information about missing values in y
print("\nMissing values in y:")
print(y.isnull().sum())

# Convert categorical columns to numerical
categorical_columns = X.select_dtypes(include=['object']).columns
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Scale the X data
X_scaled = StandardScaler().fit_transform(X)
print("\nScaled X:")
print(X_scaled)
