from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('salary_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

def model_predict(job_type, age, experience, education):
    # Prepare the input for the model
    input_data = np.array([[job_type, age, experience, education]])
    
    # Predict the salary
    prediction = model.predict(input_data)[0]
    return prediction

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    job_type = int(request.form.get('job_type'))
    age = int(request.form.get('age'))
    experience = int(request.form.get('experience'))
    education = int(request.form.get('education'))

    # Make prediction using the model
    prediction = model_predict(job_type, age, experience, education)
    
    # Return to the form with prediction
    return render_template("index.html", prediction=prediction, job_type=job_type, age=age, experience=experience, education=education)

# Create an API endpoint
@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    job_type = data['job_type']
    age = data['age']
    experience = data['experience']
    education = data['education']

    # Make prediction using the model
    prediction = model_predict(job_type, age, experience, education)
    
    # Return the prediction in JSON format
    return jsonify({
        'prediction': prediction, 
        'job_type': job_type,
        'age': age,
        'experience': experience,
        'education': education
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
