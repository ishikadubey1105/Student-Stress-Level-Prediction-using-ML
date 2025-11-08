import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')


def is_float(s):
    try:
        float(s)
        return True
    except:
        return False


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


class NumpySoftmaxRegression:
    """A minimal multinomial logistic regression implemented with numpy.

    This implementation supports fit, predict_proba, and predict. It's
    intentionally simple (batch gradient descent) to avoid scikit-learn.
    """
    def __init__(self, lr=0.1, epochs=1000, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.W = None
        self.b = None

    def fit(self, X, y):
        # X: (n_samples, n_features), y: (n_samples,) with labels 0..K-1
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        K = len(classes)

        # initialize weights
        self.W = np.zeros((K, n_features))
        self.b = np.zeros(K)

        # create one-hot
        Y = np.zeros((n_samples, K))
        for i, label in enumerate(classes):
            Y[:, i] = (y == label).astype(float)

        for epoch in range(self.epochs):
            logits = X.dot(self.W.T) + self.b
            probs = softmax(logits)
            # gradient
            gradW = (probs - Y).T.dot(X) / n_samples  # shape (K, n_features)
            gradb = np.mean(probs - Y, axis=0)
            # update
            self.W -= self.lr * gradW
            self.b -= self.lr * gradb

            if self.verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
                print(f"Epoch {epoch+1}/{self.epochs} - loss: {loss:.4f}")

    def predict_proba(self, X):
        logits = X.dot(self.W.T) + self.b
        return softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


print("Loading CSV into pandas...")
df = pd.read_csv("academic Stress level - maintainance 1.csv")
df.columns = df.columns.str.strip()

# Find target column
target = [c for c in df.columns if "stress" in c.lower()][0]
print(f"Target column: {target}")

# Drop Timestamp if present
df = df.drop(columns=["Timestamp"], errors="ignore")

# Drop missing rows
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}\n")
df = df.dropna()
print(f"Rows after removing missing values: {len(df)}")

# Prepare label encoders for categorical columns
label_encoders = {}
feature_columns = [col for col in df.columns if col != target]
print("\nEncoding categorical variables with LabelEncoder...")
for column in feature_columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
        print(f"  {column}: {len(le.classes_)} categories")

# Prepare X and y
X = df[feature_columns].values.astype(float)
y = df[target].values

print(f"\nDataset shape: {X.shape}")
unique, counts = np.unique(y, return_counts=True)
print("Target distribution:")
for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\nTraining scikit-learn LogisticRegression (pipeline with StandardScaler)...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance from logistic coefficients
try:
    clf = model.named_steps['clf']
    coefs = np.abs(clf.coef_)
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

# Convert LabelEncoders to plain mapping dicts for easier use by app/predict
label_encoders_mapping = {}
for col, le in label_encoders.items():
    label_encoders_mapping[col] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}

# Save everything needed for prediction
model_data = {
    'model': model,
    'label_encoders': label_encoders_mapping,
    'feature_columns': feature_columns,
    'target_column': target
}

pickle.dump(model_data, open("stress_model.pkl", "wb"))
print("\n✅ Model and encoders saved successfully!")
print("Saved: stress_model.pkl")