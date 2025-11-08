import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading data...")
df = pd.read_csv("academic Stress level - maintainance 1.csv")
df.columns = df.columns.str.strip()

# Find target column
target = [c for c in df.columns if "stress" in c.lower()][0]
print(f"Target column: {target}")

# Drop timestamp
df = df.drop(columns=["Timestamp"], errors="ignore")

# Handle missing values
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")
df = df.dropna()
print(f"Rows after removing missing values: {len(df)}")

# Store original column names and create label encoders
label_encoders = {}
feature_columns = [col for col in df.columns if col != target]

print("\nEncoding categorical variables...")
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        print(f"  {column}: {len(le.classes_)} categories")

# Prepare features and target
X = df[feature_columns]
y = df[target]

print(f"\nDataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train model
print("\nTraining Logistic Regression model (with standard scaling)...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance (using absolute logistic coefficients)
print("\nComputing feature importances from logistic coefficients...")
try:
    clf = model.named_steps['clf'] if isinstance(model, Pipeline) else model
    coefs = np.abs(clf.coef_)
    # For multiclass, take mean absolute coefficient across classes
    if coefs.ndim == 2:
        importances = coefs.mean(axis=0)
    else:
        importances = coefs
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    print("\nTop 5 Important Features:")
    print(feature_importance.head())
except Exception as e:
    print("Could not compute feature importances:", e)

# Save everything needed for prediction
model_data = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,
    'target_column': target
}

pickle.dump(model_data, open("stress_model.pkl", "wb"))
print("\n✅ Model and encoders saved successfully!")
print("Saved: stress_model.pkl")