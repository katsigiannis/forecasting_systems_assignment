import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
print("X_train best features using spearman feature selection", X_train_spear.columns.tolist())

#############################################
# Step 3: Dimensionality Reduction ##
#############################################
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train_spear, y_train)

# Build Autoencoder
input_dim = X_train_balanced.shape[1]
encoding_dim = 15
input_layer = Input(shape=(input_dim,))

# Structure
# encoder
encoded = Dense(64, activation='relu')(input_layer)
encoded = LeakyReLU(alpha=.1)(encoded)
encoded = Dropout(0.3)(encoded)

encoded = Dense(32, activation='relu')(encoded)
encoded = LeakyReLU(alpha=.1)(encoded)
encoded = Dropout(0.2)(encoded)
encoded_output = Dense(encoding_dim, activation='linear')(encoded)

# decoder
decoded = Dense(32, activation='relu')(encoded_output)
decoded = LeakyReLU(alpha=.1)(decoded)

decoded = Dense(64, activation='relu')(decoded)
decoded = LeakyReLU(alpha=.1)(decoded)

decoded_output = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded_output)
encoder = Model(inputs=input_layer, outputs=encoded_output)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train Autoencoder
autoencoder.fit(X_train_balanced, X_train_balanced,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
                )

# Transform Data using encoder
X_train_encoded = encoder.predict(X_train_balanced)
X_test_encoded = encoder.predict(X_test_spear)

print("X_train_encoded shape:", X_train_encoded.shape)
print("X_test_encoded shape:", X_test_encoded.shape)

#################################################################
## Step 4: Supervised Learning at UMAP manifold using catboost ##
#################################################################

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=6,
    random_seed=42,
    verbose=False
)
cat_model.fit(X_train_encoded, y_train_balanced)

y_pred = cat_model.predict(X_test_encoded)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(conf_matrix)

# PCA projection of encoded training data
pca = PCA(n_components=2, random_state=42)
X_train_encoded_pca = pca.fit_transform(X_train_encoded)

# PCA plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_train_encoded_pca[:, 0],
    y=X_train_encoded_pca[:, 1],
    hue=y_train_balanced,
    palette='Set1',
    alpha=0.7
)
plt.title("PCA projection of encoded training data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Class")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.tight_layout()
plt.show()
