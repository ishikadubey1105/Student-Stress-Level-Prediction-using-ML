import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GENERATING DETAILED DATA ANALYSIS REPORT")
print("="*70)

# Load data
df = pd.read_csv("academic Stress level - maintainance 1.csv")
df.columns = df.columns.str.strip()
target = [c for c in df.columns if "stress" in c.lower()][0]

print("\n📊 Dataset Overview:")
print(f"Total Records: {len(df)}")
print(f"Features: {len(df.columns) - 2}")  # -2 for timestamp and target

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Stress Level Distribution
plt.subplot(3, 4, 1)
stress_counts = df[target].value_counts().sort_index()
colors_stress = sns.color_palette("RdYlGn_r", len(stress_counts))
plt.bar(stress_counts.index, stress_counts.values, color=colors_stress, edgecolor='black')
plt.xlabel('Stress Level', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Stress Level Distribution', fontweight='bold')
for i, v in enumerate(stress_counts.values):
    plt.text(stress_counts.index[i], v, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Academic Stage Distribution
plt.subplot(3, 4, 2)
stage_counts = df['Your Academic Stage'].value_counts()
plt.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=sns.color_palette("pastel"))
plt.title('Academic Stage Distribution', fontweight='bold')

# 3. Study Environment
plt.subplot(3, 4, 3)
env_counts = df['Study Environment'].value_counts()
plt.barh(env_counts.index, env_counts.values, color=sns.color_palette("muted"))
plt.xlabel('Count', fontweight='bold')
plt.title('Study Environment', fontweight='bold')
for i, v in enumerate(env_counts.values):
    plt.text(v, i, str(v), va='center', fontweight='bold')

# 4. Coping Strategy
plt.subplot(3, 4, 4)
coping = df['What coping strategy you use as a student?'].value_counts()
plt.barh(range(len(coping)), coping.values, color=sns.color_palette("Set2"))
plt.yticks(range(len(coping)), ['Intellectual', 'Social', 'Emotional'])
plt.xlabel('Count', fontweight='bold')
plt.title('Coping Strategies', fontweight='bold')

# 5. Peer Pressure vs Stress
plt.subplot(3, 4, 5)
peer_stress = df.groupby('Peer pressure')[target].mean()
plt.plot(peer_stress.index, peer_stress.values, marker='o', linewidth=2, markersize=8)
plt.xlabel('Peer Pressure Level', fontweight='bold')
plt.ylabel('Average Stress Level', fontweight='bold')
plt.title('Peer Pressure vs Stress', fontweight='bold')
plt.grid(True, alpha=0.3)

# 6. Home Pressure vs Stress
plt.subplot(3, 4, 6)
home_stress = df.groupby('Academic pressure from your home')[target].mean()
plt.plot(home_stress.index, home_stress.values, marker='s', linewidth=2, markersize=8, color='orange')
plt.xlabel('Home Pressure Level', fontweight='bold')
plt.ylabel('Average Stress Level', fontweight='bold')
plt.title('Home Pressure vs Stress', fontweight='bold')
plt.grid(True, alpha=0.3)

# 7. Competition vs Stress
plt.subplot(3, 4, 7)
comp_stress = df.groupby('What would you rate the academic  competition in your student life')[target].mean()
plt.plot(comp_stress.index, comp_stress.values, marker='^', linewidth=2, markersize=8, color='red')
plt.xlabel('Competition Level', fontweight='bold')
plt.ylabel('Average Stress Level', fontweight='bold')
plt.title('Competition vs Stress', fontweight='bold')
plt.grid(True, alpha=0.3)

# 8. Bad Habits Impact
plt.subplot(3, 4, 8)
habits_stress = df.groupby('Do you have any bad habits like smoking, drinking on a daily basis?')[target].mean()
plt.bar(range(len(habits_stress)), habits_stress.values, color=['green', 'gray', 'red'])
plt.xticks(range(len(habits_stress)), ['No', 'Prefer not', 'Yes'], rotation=45)
plt.ylabel('Average Stress Level', fontweight='bold')
plt.title('Bad Habits Impact on Stress', fontweight='bold')

# 9. Correlation Heatmap
plt.subplot(3, 4, 9)
df_encoded = df.drop(['Timestamp'], axis=1, errors='ignore')
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix[[target]].sort_values(target, ascending=False), 
            annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar=False)
plt.title('Feature Correlation with Stress', fontweight='bold')

# 10. Environment & Coping Strategy Heatmap
plt.subplot(3, 4, 10)
pivot = df.pivot_table(values=target, 
                       index='Study Environment',
                       columns='What coping strategy you use as a student?',
                       aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.title('Stress by Environment & Coping', fontweight='bold', fontsize=10)
plt.xlabel('')
plt.ylabel('')

# 11. Stress by Academic Stage
plt.subplot(3, 4, 11)
stage_stress = df.groupby('Your Academic Stage')[target].agg(['mean', 'std'])
x_pos = np.arange(len(stage_stress))
plt.bar(x_pos, stage_stress['mean'], yerr=stage_stress['std'], 
        capsize=5, color=sns.color_palette("viridis", len(stage_stress)))
plt.xticks(x_pos, stage_stress.index, rotation=45, ha='right')
plt.ylabel('Stress Level', fontweight='bold')
plt.title('Stress by Academic Stage', fontweight='bold')

# 12. Overall Statistics Box
plt.subplot(3, 4, 12)
plt.axis('off')
stats_text = f"""
DATASET STATISTICS

Total Students: {len(df)}
Avg Stress: {df[target].mean():.2f}
Std Dev: {df[target].std():.2f}
Min Stress: {df[target].min()}
Max Stress: {df[target].max()}

High Stress (4-5): {len(df[df[target] >= 4])} ({len(df[df[target] >= 4])/len(df)*100:.1f}%)
Moderate (3): {len(df[df[target] == 3])} ({len(df[df[target] == 3])/len(df)*100:.1f}%)
Low Stress (1-2): {len(df[df[target] <= 2])} ({len(df[df[target] <= 2])/len(df)*100:.1f}%)
"""
plt.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('COMPREHENSIVE STUDENT STRESS ANALYSIS REPORT', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: comprehensive_analysis.png")

# Generate text report
print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)

print(f"\n1. STRESS DISTRIBUTION:")
for level, count in stress_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   Level {level}: {count} students ({percentage:.1f}%)")

print(f"\n2. HIGHEST STRESS FACTORS:")
correlations = corr_matrix[target].sort_values(ascending=False)[1:4]
for feature, corr in correlations.items():
    print(f"   • {feature}: {corr:.3f}")

print(f"\n3. ACADEMIC STAGE ANALYSIS:")
for stage, stress in df.groupby('Your Academic Stage')[target].mean().items():
    print(f"   • {stage}: {stress:.2f} avg stress")

print(f"\n4. ENVIRONMENT IMPACT:")
for env, stress in df.groupby('Study Environment')[target].mean().items():
    print(f"   • {env}: {stress:.2f} avg stress")

print("\n" + "="*70)
print("✅ Analysis complete! Check 'comprehensive_analysis.png'")
print("="*70)