import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("academic Stress level - maintainance 1.csv")
df.columns = df.columns.str.strip()

target = [c for c in df.columns if "stress" in c.lower()][0]
df = df.drop(columns=["Timestamp"], errors="ignore")

label = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label.fit_transform(df[column])

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully ✅")

pickle.dump(model, open("stress_model.pkl", "wb"))
