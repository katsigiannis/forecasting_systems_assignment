import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

#################################################
# Step 1: Load Dataset - Split - Preprocessing ##
#################################################

# Load dataset from disk
CREDIT_SCORE_DATASET_PATH = "../dataset/credit_score.csv"
df = pd.read_csv(CREDIT_SCORE_DATASET_PATH)
print("Dataset dimensions:", df.shape)
print("Column names:", df.columns.tolist())

X = df.drop(columns=['DEFAULT', 'CUST_ID'])
y = df['DEFAULT']


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display information about missing values in X
print("\nMissing values in X:")
print(X.isnull().sum())
print("\nTotal missing values:", X.isnull().sum().sum())

# Display information about missing values in y
print("\nMissing values in y:")
print(y.isnull().sum())

# Convert categorical columns to numerical
# Warning: For better stability, not modify a view but a full copy
X_train = X_train.copy()
X_test = X_test.copy()

# Warning: we are labeling using X_train and not X
categorical_columns = X_train.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
X_test[categorical_columns] = encoder.transform(X_test[categorical_columns])

# Scale the data
# Warning: we are using fit_transform and transform for X_train and X_test correspondingly
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

#############################################
# Step 2: Feature selection using Spearman ##
#############################################

X_train_scaled['DEFAULT'] = y_train.values

spearman_correlation = X_train_scaled.corr(method='spearman')['DEFAULT'].drop('DEFAULT')
selected_features = spearman_correlation[abs(spearman_correlation) > 0.2].index.tolist()
print("Selected features:", selected_features)
print("Number of selected features ", len(selected_features))

X_train_spear = X_train_scaled[selected_features]
X_test_spear = X_test_scaled[selected_features]

#############################################
# Step 3: UMAP Dimensionality Reduction ##
#############################################

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, n_components=2)
X_train_umap = umap_model.fit_transform(X_train_spear)
X_test_umap = umap_model.transform(X_test_spear)

print("X_train_umap shape:", X_train_umap.shape)
print("X_test_umap shape:", X_test_umap.shape)

# Plotting of the first 2 dimensions
print("UMAP plot of the first 2 dimensions")
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='Spectral')
plt.colorbar()
plt.title("UMAP plot of the first 2 dimensions")
plt.show()

##################################################
## Step 4: Supervised Learning at UMAP manifold ##
##################################################
