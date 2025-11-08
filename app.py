from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model_data = pickle.load(open("stress_model.pkl", "rb"))
model = model_data['model']
label_encoders = model_data['label_encoders']
feature_columns = model_data['feature_columns']

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
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col in df.columns:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
        
        # Ensure correct column order
        df = df[feature_columns]
        
        # Predict
        prediction = int(model.predict(df)[0])
        probability = model.predict_proba(df)[0]
        confidence = float(max(probability) * 100)
        
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
            'info': stress_info[prediction]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)