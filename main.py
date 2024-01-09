from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model


app = Flask(__name__)

# Load diabetes model
with open('models/diabetes.sav', 'rb') as file:
    diabetes_model = pickle.load(file)

# Load liver model
with open('models/liver.pkl', 'rb') as file:
    liver_model = pickle.load(file)

# Load heart model
with open('models/heart.pkl', 'rb') as file:
    heart_model = pickle.load(file)

# Load kidney model
with open('models/kidney.pkl', 'rb') as file:
    kidney_model = pickle.load(file)


# Load breast cancer model
with open('models/breast_cancer.pkl', 'rb') as file:
    breast_cancer_model = pickle.load(file)

# Other routes...

# Route to display the breast cancer prediction page
@app.route('/breast_cancer')
def breast_cancer():
    return render_template('breast_cancer.html')

# Route for breast cancer predictions
@app.route('/predict_breast_cancer', methods=['POST'])
def predict_breast_cancer():
    # Extract data from form fields for breast cancer prediction
    values = [
        float(request.form['radius_mean']),
        float(request.form['texture_mean']),
        float(request.form['perimeter_mean']),
        float(request.form['area_mean']),
        float(request.form['smoothness_mean']),
        float(request.form['compactness_mean']),
        float(request.form['concavity_mean']),
        float(request.form['concave_points_mean']),
        float(request.form['symmetry_mean']),
        float(request.form['radius_se']),
        float(request.form['perimeter_se']),
        float(request.form['area_se']),
        float(request.form['compactness_se']),
        float(request.form['concavity_se']),
        float(request.form['concave_points_se']),
        float(request.form['fractal_dimension_se']),
        float(request.form['radius_worst']),
        float(request.form['texture_worst']),
        float(request.form['perimeter_worst']),
        float(request.form['area_worst']),
        float(request.form['smoothness_worst']),
        float(request.form['compactness_worst']),
        float(request.form['concavity_worst']),
        float(request.form['concave_points_worst']),
        float(request.form['symmetry_worst']),
        float(request.form['fractal_dimension_worst']),
        float(request.form['frt']),
        float(request.form['ftal']),
        float(request.form['imensio']),
        float(request.form['nsio'])

    ]
    # Reshape values for model prediction
    values = np.asarray(values).reshape(1, -1)
    # Make a prediction
    prediction = breast_cancer_model.predict(values)
    # Format prediction for rendering
    prediction = str(prediction[0])
    # Return the result to the template
    return render_template('breast_cancer.html', prediction=prediction)


@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

# Route for kidney predictions
@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    # Extract data from form fields for kidney prediction
    values = [
        int(request.form['Age']),
        int(request.form['bp']),
        float(request.form['al']),
        float(request.form['su']),
        float(request.form['rbc']),
        float(request.form['pc']),
        float(request.form['pcc']),
        float(request.form['ba']),
        float(request.form['bgr']),
        float(request.form['bu']),
        float(request.form['sc']),
        float(request.form['pot']),
        #float(request.form['wc']),
        #float(request.form['htn']),
        #float(request.form['dm']),
        #float(request.form['cad']),
        #float(request.form['pe']),
        #float(request.form['ane'])
    ]
    # Reshape values for model prediction
    values = np.asarray(values).reshape(1, -1)
    # Make a prediction
    prediction = kidney_model.predict(values)
    # Format prediction for rendering
    prediction = str(prediction[0])
    # Return the result to the template
    return render_template('kidney.html', prediction=prediction)



@app.route('/heart')
def heart():
    return render_template('heart.html')

# Route for heart predictions
@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    # Extract data from form fields for heart prediction
    values = [
        int(request.form['Age']),
        int(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        #float(request.form['thal'])
    ]
    # Reshape values for model prediction
    values = np.asarray(values).reshape(1, -1)
    # Make a prediction
    prediction = heart_model.predict(values)
    # Format prediction for rendering
    prediction = str(prediction[0])
    # Return the result to the template
    return render_template('heart.html', prediction=prediction)

# Home page with links to prediction models
@app.route('/')
def home():
    return render_template('index.html')

# Diabetes prediction page
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

# Diabetes prediction handling
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    # Extract data from form fields and convert to appropriate numeric types
    values = [
        int(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        int(request.form['age'])
    ]
    # Reshape values for model prediction
    values = np.asarray(values).reshape(1, -1)
    # Make a prediction
    prediction = diabetes_model.predict(values)
    # Format prediction for rendering
    prediction = str(prediction[0])
    # Return the result to the template
    return render_template('diabetes.html', prediction=prediction)

# Liver prediction page
@app.route('/liver')
def liver():
    return render_template('liver.html')

# Liver prediction handling
@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    # Extract data from form fields for liver prediction
    values = [
        int(request.form['Age']),
        int(request.form['Gender']),
        float(request.form['Total_Bilirubin']),
        float(request.form['Direct_Bilirubin']),
        float(request.form['Alkaline_Phosphotase']),
        float(request.form['Alamine_Aminotransferase']),
        float(request.form['Aspartate_Aminotransferase']),
        float(request.form['Total_Protiens']),
        float(request.form['Albumin']),
        float(request.form['Albumin_and_Globulin_Ratio'])
    ]
    # Reshape values for model prediction
    values = np.asarray(values).reshape(1, -1)
    # Make a prediction
    prediction = liver_model.predict(values)
    # Format prediction for rendering
    prediction = str(prediction[0])
    # Return the result to the template
    return render_template('liver.html', prediction=prediction)

# Malaria prediction route
@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
                return render_template('malaria_predict.html', pred=pred)
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)
    return render_template('malaria.html')

# Pneumonia prediction route
@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
                return render_template('pneumonia_predict.html', pred=pred)
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True)
