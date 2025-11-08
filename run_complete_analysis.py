"""
Complete ML Analysis Runner
Runs all analysis scripts and generates comprehensive reports
"""

import os
import sys

print("="*70)
print("STUDENT STRESS PREDICTION - COMPLETE ANALYSIS SUITE")
print("="*70)

scripts = [
    ("Data Analysis Report", "generate_report.py"),
    ("Algorithm Comparison", "compare_algorithms.py")
]

print("\nThis will generate:")
print("  ✓ Comprehensive data analysis report")
print("  ✓ Algorithm comparison visualizations")
print("  ✓ Performance metrics and charts")
print("  ✓ Feature importance analysis")

input("\nPress Enter to start analysis...")

for name, script in scripts:
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print('='*70)
    
    try:
        exec(open(script).read())
    except FileNotFoundError:
        print(f"❌ Error: {script} not found!")
        print(f"Please make sure {script} is in the same folder.")
    except Exception as e:
        print(f"❌ Error running {script}: {e}")
    
    print()

print("\n" + "="*70)
print("🎉 ALL ANALYSIS COMPLETE!")
print("="*70)

print("\nGenerated Files:")
print("  📊 comprehensive_analysis.png - Data insights")
print("  📈 algorithm_comparison.png - Accuracy comparison")
print("  📉 cv_accuracy_comparison.png - Cross-validation results")
print("  🎯 top3_confusion_matrices.png - Top model performance")
print("  🔥 performance_heatmap.png - Algorithm metrics")
print("  🎨 radar_comparison.png - Top 5 algorithms")
print("  🌟 feature_importance.png - Key factors")

print("\n" + "="*70)
print("Use these visualizations for your project presentation!")
print("="*70)