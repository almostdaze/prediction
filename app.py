from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('salary_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a mapping for job types and education levels
job_type_mapping = {
    'Software Engineer': 0,
    'Data Scientist': 1,
    'Doctor': 2,
    'Teacher': 3,
    'Accountant': 4,
    'Engineer': 5
}

education_mapping = {
    'Bachelor': 0,
    'Master': 1,
    'PhD': 2
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    job_type = request.form.get('job_type')
    age = request.form.get('age')
    experience = request.form.get('experience')
    education = request.form.get('education')

    # Convert job_type and education to numerical values using the mappings
    job_type_num = job_type_mapping[job_type]
    education_num = education_mapping[education]

    # Prepare input for the model (in the correct order and format)
    input_data = np.array([[job_type_num, int(age), int(experience), education_num]])

    # Make a prediction
    salary_prediction = model.predict(input_data)[0]

    # Convert salary to an integer to remove decimals
    salary_prediction = int(salary_prediction)

    # Pass the form data back to the template
    return render_template("index.html", prediction=salary_prediction, 
                           job_type=job_type, age=age, 
                           experience=experience, education=education)

if __name__ == "__main__":
    app.run(debug=True)