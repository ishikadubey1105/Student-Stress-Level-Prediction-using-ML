from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model_data = pickle.load(open("stress_model.pkl", "rb"))
model = model_data['model']
label_encoders = model_data.get('label_encoders', {})
feature_columns = model_data['feature_columns']

# If label_encoders were saved as sklearn LabelEncoder objects, convert to mapping dicts
for k, v in list(label_encoders.items()):
    try:
        if hasattr(v, 'classes_'):
            mapping = {str(cls): int(i) for i, cls in enumerate(v.classes_)}
            label_encoders[k] = mapping
    except Exception:
        pass


def _encode_input(data):
    x = []
    for col in feature_columns:
        val = data.get(col, "")
        if col in label_encoders:
            x.append(float(label_encoders[col].get(str(val), 0)))
        else:
            x.append(float(val))
    return np.array(x, dtype=float).reshape(1, -1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Your Academic Stage': request.form['academic_stage'],
            'Peer pressure': int(request.form['peer_pressure']),
            'Academic pressure from your home': int(request.form['home_pressure']),
            'Study Environment': request.form['study_env'],
            'What coping strategy you use as a student?': request.form['coping_strategy'],
            'Do you have any bad habits like smoking, drinking on a daily basis?': request.form['bad_habits'],
            'What would you rate the academic  competition in your student life': int(request.form['competition'])
        }

        X = _encode_input(data)
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        idx_to_target = model_data.get('idx_to_target')
        if idx_to_target is not None:
            prediction = int(idx_to_target.get(pred_idx, pred_idx))
        else:
            try:
                prediction = int(model.predict(X)[0])
            except Exception:
                prediction = int(pred_idx)
        confidence = float(max(probs) * 100)

        # Stress interpretation
        stress_info = {
            1: {"level": "Very Low", "color": "#22c55e", "advice": "Excellent! Keep maintaining your current balance."},
            2: {"level": "Low", "color": "#84cc16", "advice": "You're managing well. Continue your good habits."},
            3: {"level": "Moderate", "color": "#eab308", "advice": "Normal academic pressure. Stay organized."},
            4: {"level": "High", "color": "#f97316", "advice": "Consider stress management techniques and reach out for support."},
            5: {"level": "Very High", "color": "#ef4444", "advice": "Important: Please seek support from counselors or trusted adults."}
        }

        result = {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'info': stress_info.get(prediction, {})
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)