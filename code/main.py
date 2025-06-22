import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.manifold import TSNE
from catboost import CatBoostClassifier
from scipy.stats import pearsonr

##################################################
# Step 0: We are creating some helpful functions #
##################################################

def modify_categorical_cols(x):
    """
    Converts categorical columns in a DataFrame to numerical values using OrdinalEncoder.

    Args:
        x (pd.DataFrame): Input DataFrame containing categorical columns

    Returns:
        pd.DataFrame: DataFrame with categorical columns converted to numerical values
    """

    categorical_columns = x.select_dtypes(include=['object']).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    x[categorical_columns] = encoder.fit_transform(x[categorical_columns])
    return x

def scale_dataset(x):
    """
    Scales numerical features in a DataFrame using StandardScaler.

    Args:
        x (pd.DataFrame): Input DataFrame containing numerical features to be scaled

    Returns:
        pd.DataFrame: DataFrame with scaled numerical features, maintaining original column names
    """

    # NOTE: we are using fit_transform and transform for X_train and X_test correspondingly
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

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

# NOTE: For better stability, not modify a view but a full copy
X = X.copy()

# Categorical to numerical labels to X
X = modify_categorical_cols(X)

# Scale dataset
X_scaled = scale_dataset(X)
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

#####################################################
# Step 5: Feature Selection from combined Criteria ##
#####################################################
# XXX: Use the scatter graph to deside the number of the features.
top_n = 5

# Take importance col with "Feature" indices
importance = importance_df.set_index("Feature")["Importance"]
# Take pearson correlation col with "Feature" indices
correlation = correlation_df.set_index("Feature")["Pearson"]

ranked_importance = importance.rank(ascending=False)
ranked_correlation = correlation.rank(ascending=False)
print("Ranked correlation:")
print(ranked_correlation)
print("Ranked ranked_importance:")
print(ranked_importance)

combined_rank = (ranked_correlation+ranked_importance)/2
print("Combined rank:")
print(combined_rank)

combined_rank_df = combined_rank.sort_values().reset_index()
combined_rank_df.columns = ["Feature", "AverageRank"]
print("Top 5 features:")
print(combined_rank_df.head(top_n))

# n-top feature
selected_features = combined_rank_df.head(top_n)["Feature"].tolist()
print("Selected features:", selected_features)

# NOTE: We what to rescale and retrain the catboost algorithm
# using the ->->initial<-<- dataset
X_selected = df[selected_features]
y_final = df['DEFAULT']


# Split data to train and test sets

# modify categorical cols to numerical

# scale data


############################################
# Step 6: Create a new catboost classifier #
#     and retrain with df_selected Dataset #
############################################
