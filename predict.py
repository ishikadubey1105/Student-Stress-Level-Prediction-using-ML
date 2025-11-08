import pickle
import pandas as pd
import numpy as np

# Load the trained model and encoders
print("Loading model...")
model_data = pickle.load(open("stress_model.pkl", "rb"))

model = model_data['model']
label_encoders = model_data['label_encoders']
feature_columns = model_data['feature_columns']
target_column = model_data['target_column']

print("Model loaded successfully!\n")

def predict_stress(data_dict):
    """
    Predict stress level from input data
    
    Args:
        data_dict: Dictionary with feature names and values
    
    Returns:
        Predicted stress level (1-5)
    """
    # Create DataFrame from input
    df = pd.DataFrame([data_dict])
    
    # Encode categorical variables
    for col in df.columns:
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError as e:
                print(f"Warning: Unknown category in {col}")
                df[col] = 0
    
    # Ensure correct column order
    df = df[feature_columns]
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    return prediction, probability

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("STUDENT STRESS LEVEL PREDICTION")
    print("="*60)
    
    # Example student data
    student_data = {
        'Your Academic Stage': 'undergraduate',
        'Peer pressure': 4,
        'Academic pressure from your home': 5,
        'Study Environment': 'disrupted',
        'What coping strategy you use as a student?': 'Analyze the situation and handle it with intellect',
        'Do you have any bad habits like smoking, drinking on a daily basis?': 'No',
        'What would you rate the academic  competition in your student life': 4
    }
    
    print("\nInput Data:")
    for key, value in student_data.items():
        print(f"  {key}: {value}")
    
    prediction, probabilities = predict_stress(student_data)
    
    print(f"\n{'='*60}")
    print(f"PREDICTED STRESS LEVEL: {prediction} / 5")
    print(f"{'='*60}")
    
    print(f"\nConfidence Distribution:")
    for i, prob in enumerate(probabilities, 1):
        bar = '█' * int(prob * 50)
        print(f"  Level {i}: {bar} {prob*100:.1f}%")
    
    # Stress level interpretation
    stress_labels = {
        1: "Very Low Stress - Excellent mental state",
        2: "Low Stress - Managing well",
        3: "Moderate Stress - Normal academic pressure",
        4: "High Stress - Consider stress management",
        5: "Very High Stress - Seek support immediately"
    }
    
    print(f"\nInterpretation: {stress_labels.get(prediction, 'Unknown')}")
    
    # Interactive prediction
    print("\n" + "="*60)
    print("Want to make another prediction? Edit the student_data dictionary")
    print("and run the script again!")
    print("="*60)