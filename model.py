import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
from typing import Tuple
warnings.filterwarnings("ignore")

# ---- Config ----
CSV_FILENAME = "academic Stress level - maintainance 1.csv"  # adjust if different
MODEL_FILENAME = "stress_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Put your dataset CSV in the repo root or update the path.")
    df = pd.read_csv(csv_path)
    return df

def basic_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    # drop completely empty rows
    df = df.dropna(how="all")
    df = df.dropna()  # simple: drop rows with any NA; change if you want imputation

    # identify target column - many datasets call it 'stress' or 'Stress level'
    # Let's try to detect the target automatically:
    possible_targets = [c for c in df.columns if "stress" in c.lower() or "level" in c.lower()]
    if len(possible_targets) == 0:
        # fallback: assume last column is target
        target_col = df.columns[-1]
    else:
        # prefer exact matches like 'Stress' or 'stress level'
        target_col = possible_targets[0]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If y is non-numeric categories, label encode
    if y.dtype == object or not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
        # save mapping for future predictions if you want:
        mapping = dict(enumerate(le.classes_))
        print("Saved target mapping (index -> class):", mapping)
    else:
        # if numeric but floats representing classes, convert to int
        y = y.astype(int)

    # For simplicity, convert any non-numeric features with get_dummies
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def train_and_save(X: pd.DataFrame, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ])

    param_grid = {
        "clf__penalty": ["l2"],  # 'l1' requires solver='liblinear' or 'saga' - we'll enable 'saga' below
        "clf__C": [0.01, 0.1, 1, 10, 100],
        # if you want to try l1 too:
        # "clf__penalty": ["l1", "l2"],
        # "clf__solver": ["saga"],  # saga supports l1 and l2 for LogisticRegression
    }

    # If you prefer l1/l2 search uncomment/adjust above and set solver to 'saga'
    # For robust search, use solver 'saga' (works with l1 and l2), but can be slower.
    # Example: pipeline.named_steps['clf'].set_params(solver='saga')

    # Use GridSearchCV
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("\nBest params:", grid.best_params_)
    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Cross-validated accuracy on full data
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"\n5-fold cross-validated accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save final pipeline
    joblib.dump(best_model, MODEL_FILENAME)
    print(f"\nSaved trained model to {MODEL_FILENAME}")

    return best_model

def main():
    df = load_data(CSV_FILENAME)
    print("Loaded dataset with shape:", df.shape)
    X, y = basic_preprocess(df)
    print("Feature matrix shape:", X.shape, "Target shape:", y.shape)
    trained = train_and_save(X, y)

if __name__ == "__main__":
    main()
