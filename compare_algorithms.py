import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import different algorithms
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("STUDENT STRESS PREDICTION - ML ALGORITHM COMPARISON")
print("="*70)

# Load and preprocess data
print("\n📁 Loading dataset...")
df = pd.read_csv("academic Stress level - maintainance 1.csv")
df.columns = df.columns.str.strip()

target = [c for c in df.columns if "stress" in c.lower()][0]
df = df.drop(columns=["Timestamp"], errors="ignore")
df = df.dropna()

print(f"Dataset shape: {df.shape}")
print(f"Target column: {target}")

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Prepare features and target
feature_columns = [col for col in df.columns if col != target]
X = df[feature_columns]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Define algorithms to compare
algorithms = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

# Store results
results = []
confusion_matrices = {}

print("\n" + "="*70)
print("TRAINING AND EVALUATING ALGORITHMS...")
print("="*70 + "\n")

# Train and evaluate each algorithm
for name, model in algorithms.items():
    print(f"Training {name}...", end=" ")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results.append({
        'Algorithm': name,
        'Accuracy': accuracy * 100,
        'CV Mean': cv_mean * 100,
        'CV Std': cv_std * 100
    })
    
    # Store confusion matrix
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    print(f"✓ Accuracy: {accuracy*100:.2f}%")

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# ==================== VISUALIZATIONS ====================

# 1. Accuracy Comparison Bar Chart
plt.figure(figsize=(14, 6))
colors = sns.color_palette("husl", len(results_df))
bars = plt.bar(results_df['Algorithm'], results_df['Accuracy'], color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('ML Algorithm Accuracy Comparison\nStudent Stress Prediction', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: algorithm_comparison.png")

# 2. Accuracy with Error Bars (Cross-Validation)
plt.figure(figsize=(14, 6))
x_pos = np.arange(len(results_df))
plt.bar(x_pos, results_df['CV Mean'], yerr=results_df['CV Std'], 
        capsize=5, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
plt.ylabel('Cross-Validation Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Cross-Validation Accuracy with Standard Deviation', fontsize=14, fontweight='bold')
plt.xticks(x_pos, results_df['Algorithm'], rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('cv_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: cv_accuracy_comparison.png")

# 3. Top 3 Algorithms - Detailed Comparison
plt.figure(figsize=(15, 5))
top_3 = results_df.head(3)

for idx, (i, row) in enumerate(top_3.iterrows(), 1):
    plt.subplot(1, 3, idx)
    model_name = row['Algorithm']
    cm = confusion_matrices[model_name]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title(f'{model_name}\nAccuracy: {row["Accuracy"]:.2f}%', fontweight='bold')
    plt.xlabel('Predicted Stress Level')
    plt.ylabel('Actual Stress Level')

plt.tight_layout()
plt.savefig('top3_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Saved: top3_confusion_matrices.png")

# 4. Algorithm Performance Metrics Heatmap
plt.figure(figsize=(10, 8))
metrics_df = results_df.set_index('Algorithm')[['Accuracy', 'CV Mean']]
sns.heatmap(metrics_df, annot=True, fmt='.2f', cmap='RdYlGn', center=70,
            cbar_kws={'label': 'Score (%)'}, linewidths=1, linecolor='black')
plt.title('Algorithm Performance Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: performance_heatmap.png")

# 5. Radar Chart for Top 5 Algorithms
from math import pi

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

top_5 = results_df.head(5)
categories = list(top_5['Algorithm'])
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for metric in ['Accuracy', 'CV Mean']:
    values = top_5[metric].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=metric)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 100)
ax.set_title('Top 5 Algorithms - Radar Comparison', fontweight='bold', size=14, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: radar_comparison.png")

# 6. Feature Importance for Best Model
best_model_name = results_df.iloc[0]['Algorithm']
best_model = algorithms[best_model_name]

if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    colors_fi = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
             color=colors_fi, edgecolor='black', linewidth=1)
    plt.xlabel('Importance Score', fontweight='bold')
    plt.ylabel('Features', fontweight='bold')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: feature_importance.png")

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)
print(f"\n🏆 Best Algorithm: {results_df.iloc[0]['Algorithm']}")
print(f"📊 Best Accuracy: {results_df.iloc[0]['Accuracy']:.2f}%")
print(f"\n📁 All visualizations saved successfully!")
print("\nGenerated files:")
print("  1. algorithm_comparison.png")
print("  2. cv_accuracy_comparison.png")
print("  3. top3_confusion_matrices.png")
print("  4. performance_heatmap.png")
print("  5. radar_comparison.png")
print("  6. feature_importance.png")
print("="*70)