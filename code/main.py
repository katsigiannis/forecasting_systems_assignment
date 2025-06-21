import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.manifold import TSNE
from catboost import CatBoostClassifier
from scipy.stats import pearsonr

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

# Display information about missing values in X
print("\nMissing values in X:")
print(X.isnull().sum())
print("\nTotal missing values:", X.isnull().sum().sum())

# Display information about missing values in y
print("\nMissing values in y:")
print(y.isnull().sum())

# Convert categorical columns to numerical
# Warning: For better stability, not modify a view but a full copy
X = X.copy()

# Warning: we are labeling using X_train and not X
categorical_columns = X.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = encoder.fit_transform(X[categorical_columns])

# Scale the data
# Warning: we are using fit_transform and transform for X_train and X_test correspondingly
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("X_train_scaled shape:", X_scaled.shape)

###########################
# Step 2: Apply catboost ##
###########################

# Initialize and train CatBoost model
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    random_seed=42,
    verbose=False
)

# Fit the model
catboost_model.fit(X_scaled, y)

# Extract the feature importance
feature_importance = catboost_model.get_feature_importance()

importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": feature_importance
    }
).sort_values(by="Importance", ascending=False)

print("Feature importance:")
print(importance_df)

#######################################################
# Step 3: Correlation analysis on importance features #
#######################################################

# Correlation between features and target

correlation_scores = []
p_values = []

for feature in X.columns:
   x_col = X_scaled[feature]
   r, p =pearsonr(x_col, y)
   correlation_scores.append(abs(r))
   p_values.append(p)

# Implement into a DataFrame
correlation_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Pearson": correlation_scores,
        "P-value": p_values
    }
).sort_values(by="Pearson", ascending=False)

print("Correlation analysis:")
print(correlation_df)

########################################
# Step 4: Manifold Learning with t-SNE #
########################################

# Create the tsne model fit to data
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize the model after learning

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='k')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.colorbar(scatter, label = 'Default (0=No, 1=Yes)')
plt.show()

