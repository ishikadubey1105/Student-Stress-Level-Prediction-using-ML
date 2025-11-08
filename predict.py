import pickle
import numpy as np

# Load the trained model and encoders
print("Loading model...")
model_data = pickle.load(open("stress_model.pkl", "rb"))

model = model_data['model']
label_encoders = model_data.get('label_encoders', {})
feature_columns = model_data['feature_columns']

# If label_encoders were saved as sklearn LabelEncoder objects, convert to mapping dicts
for k, v in list(label_encoders.items()):
    try:
        # LabelEncoder has attribute classes_
        if hasattr(v, 'classes_'):
            mapping = {str(cls): int(i) for i, cls in enumerate(v.classes_)}
            label_encoders[k] = mapping
    except Exception:
        pass

print("Model loaded successfully!\n")


def _encode_and_order(data_dict):
    x = []
    for col in feature_columns:
        val = data_dict.get(col, "")
        if col in label_encoders:
            # map unknown categories to 0
            x.append(float(label_encoders[col].get(str(val), 0)))
        else:
            x.append(float(val))
    return np.array(x, dtype=float).reshape(1, -1)


def predict_stress(data_dict):
    """Predict stress level from input data (dict)-> (prediction, probabilities)

    Returns prediction as original target value (e.g., 1-5) and probability vector.
    """
    X = _encode_and_order(data_dict)
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    # Model saved targets are numeric (e.g., 1-5). We saved mapping in model_data->idx_to_target when using numpy trainer,
    # but for scikit-learn we keep original target labels in the model. So attempt to infer original target:
    idx_to_target = model_data.get('idx_to_target')
    if idx_to_target is not None:
        prediction = idx_to_target.get(pred_idx, pred_idx)
    else:
        # If no mapping, the classifier likely predicts original labels directly
        try:
            prediction = int(model.predict(X)[0])
        except Exception:
            prediction = int(pred_idx)
    return int(prediction), probs

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