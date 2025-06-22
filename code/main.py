import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE
from catboost import CatBoostClassifier
from scipy.stats import pearsonr

##################################################
# Step 0: We are creating some helpful functions #
##################################################

def modify_categorical_cols(x, t=pd.DataFrame()):
    """
    Converts categorical columns in a DataFrame to numerical values using OrdinalEncoder.

    Args:
        x (pd.DataFrame): Input DataFrame containing categorical columns
        t (pd.DataFrame): Optional DataFrame to encode using same encoder as x

    Returns:
        pd.DataFrame: DataFrame with categorical columns converted to numerical values
    """

    categorical_columns = x.select_dtypes(include=['object']).columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    x[categorical_columns] = encoder.fit_transform(x[categorical_columns])
    if not t.empty:
        t[categorical_columns] = encoder.transform(t[categorical_columns])
    return x,t

def scale_dataset(x, t=pd.DataFrame()):
    """
    Scales numerical features in a DataFrame using StandardScaler.

    Args:
        x (pd.DataFrame): Input DataFrame containing numerical features to be scaled
        t (pd.DataFrame): Optional DataFrame to scale using the same scaler as x

    Returns:
        pd.DataFrame: DataFrame with scaled numerical features, maintaining original column names
    """

    # NOTE: we are using fit_transform and transform for X_train and X_test correspondingly
    t_scal = pd.DataFrame()
    scaler = StandardScaler()
    x_scal = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    if not t.empty:
        t_scal = pd.DataFrame(scaler.transform(t), columns=t.columns)
    return x_scal, t_scal

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
X, _ = modify_categorical_cols(X)

# Scale dataset
X_scaled, _ = scale_dataset(X)
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
top_n = 10

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
X_final = df[selected_features]
y_final = df['DEFAULT']

############################################
# Step 6: Create a new catboost classifier #
#     and retrain with df_selected Dataset #
############################################
# Split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42
)

# encode categorical cols to numerical
# NOTE: Encode train only categorical data
# the test data will have the same index -1

X_train_encoded, X_test_encoded = modify_categorical_cols(X_train, X_test)
print("X_train_encoded shape:", X_train_encoded.shape)
print("X_test_encoded shape:", X_test_encoded.shape)

# scale data
# NOTE: The test data will have scales accordingly to train data
X_train_scaled, X_test_scaled = scale_dataset(X_train_encoded, X_test_encoded)
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("X_test_scaled", X_test_scaled)

# Create the new catboost model
final_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    random_seed=42,
    verbose=False
)

# Train the model
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)
y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]

##############################
# Step 7: Results and plots ##
##############################
# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_proba))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plots

# Confusion matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[0, 1],
            yticklabels=[0, 1],
            linewidths=1,
            linecolor='black'
            )
plt.title("Confusion Matrix Heatmap")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Feature Importances
importances = final_model.get_feature_importance()
feature_labels = selected_features

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_labels, orient='h')
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()