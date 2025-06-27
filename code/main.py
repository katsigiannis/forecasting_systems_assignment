import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.manifold import Isomap

#################################################
# Step 1: Load Dataset - Split - Preprocessing ##
#################################################

# Load dataset from disk
CREDIT_SCORE_DATASET_PATH = "../dataset/credit_score.csv"
df = pd.read_csv(CREDIT_SCORE_DATASET_PATH)
print("Dataset dimensions:", df.shape)
print("Column names:", df.columns.tolist())

X = df.drop(columns=['DEFAULT', 'CUST_ID', 'CREDIT_SCORE'])
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
print("X_train best features using spearman feature selection", X_train_spear.columns.tolist())

#############################################
# Step 3: UMAP Dimensionality Reduction ##
#############################################
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train_spear, y_train)

isomap_model = Isomap(n_neighbors=10, n_components=10)
X_train_isomap = isomap_model.fit_transform(X_train_balanced)
X_test_isomap = isomap_model.transform(X_test_spear)

print("X_train_isomap shape:", X_train_isomap.shape)
print("X_test_isomap shape:", X_test_isomap.shape)

#################################################################
## Step 4: Supervised Learning at UMAP manifold using catboost ##
#################################################################

cat_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=False)
cat_model.fit(X_train_isomap, y_train_balanced)

y_pred = cat_model.predict(X_test_isomap)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))