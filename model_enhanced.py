import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# Helper to identify numeric columns
def get_numeric_columns(df):
    return df.select_dtypes(include=['int64', 'float64']).columns

# Helper to identify categorical columns
def get_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns

print("Loading CSV into pandas...")
df = pd.read_csv("academic Stress level - maintainance 1.csv")
df.columns = df.columns.str.strip()

# Find target column
target = [c for c in df.columns if "stress" in c.lower()][0]
print(f"Target column: {target}")

# Drop Timestamp and handle missing values
df = df.drop(columns=["Timestamp"], errors="ignore").dropna()
print(f"Rows after cleaning: {len(df)}")

# Identify feature types
numeric_features = [col for col in get_numeric_columns(df) if col != target]
categorical_features = list(get_categorical_columns(df))

print("\nFeature types:")
print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # This drops the target column
)

# Create a preprocessing and modeling pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    ))
])

# Split data
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define hyperparameter search space
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__solver': ['lbfgs', 'newton-cg'],
    'classifier__max_iter': [2000]
}

print("\nPerforming grid search for hyperparameter tuning...")
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Use best model for final evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get feature names after preprocessing
def get_feature_names(column_transformer):
    feature_names = []
    
    for name, pipe, features in column_transformer.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            # Get feature names for categorical variables after one-hot encoding
            cats = pipe.named_steps['onehot'].get_feature_names_out(features)
            feature_names.extend(cats)
    
    return feature_names

# Extract feature importance
feature_names = get_feature_names(best_model.named_steps['preprocessor'])
coef = best_model.named_steps['classifier'].coef_
importances = np.abs(coef).mean(axis=0)  # Average across classes

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Store mapping of categorical values for prediction
cat_mappings = {}
for feature in categorical_features:
    values = sorted(df[feature].unique())
    cat_mappings[feature] = {v: i for i, v in enumerate(values)}

model_data = {
    'model': best_model,
    'label_encoders': cat_mappings,  # Store as simple mappings for prediction
    'feature_columns': list(X.columns),
    'target_column': target,
    'categorical_features': categorical_features,
    'numeric_features': numeric_features
}

with open("stress_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\n✅ Model and encoders saved successfully!")
print("Saved: stress_model.pkl")

# Print model summary
print("\nModel Summary:")
print(f"- Features: {len(feature_names)} total")
print(f"  - {len(numeric_features)} numeric features")
print(f"  - {len(categorical_features)} categorical features (one-hot encoded)")
print(f"- Best hyperparameters: C={grid_search.best_params_['classifier__C']}, solver={grid_search.best_params_['classifier__solver']}")
print(f"- Cross-validation accuracy: {grid_search.best_score_:.3f}")
print(f"- Test set accuracy: {accuracy:.3f}")