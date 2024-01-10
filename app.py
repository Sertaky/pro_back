from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model

app = Flask(__name__)
api = Api(app)

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

# Load malaria model
malaria_model = load_model("models/malaria.h5")

# Load pneumonia model
pneumonia_model = load_model("models/pneumonia.h5")

# Create reqparse object for each prediction endpoint
diabetes_parser = reqparse.RequestParser()
diabetes_parser.add_argument('pregnancies', type=int)
diabetes_parser.add_argument('glucose', type=float)
diabetes_parser.add_argument('bloodpressure', type=float)
diabetes_parser.add_argument('skinthickness', type=float)
diabetes_parser.add_argument('insulin', type=float)
diabetes_parser.add_argument('bmi', type=float)
diabetes_parser.add_argument('dpf', type=float)
diabetes_parser.add_argument('age', type=int)

liver_parser = reqparse.RequestParser()
liver_parser.add_argument('Age', type=int)
liver_parser.add_argument('Gender', type=int)
liver_parser.add_argument('Total_Bilirubin', type=float)
liver_parser.add_argument('Direct_Bilirubin', type=float)
liver_parser.add_argument('Alkaline_Phosphotase', type=float)
liver_parser.add_argument('Alamine_Aminotransferase', type=float)
liver_parser.add_argument('Aspartate_Aminotransferase', type=float)
liver_parser.add_argument('Total_Protiens', type=float)
liver_parser.add_argument('Albumin', type=float)
liver_parser.add_argument('Albumin_and_Globulin_Ratio', type=float)

breast_cancer_parser = reqparse.RequestParser()
breast_cancer_parser.add_argument('radius_mean', type=float)
breast_cancer_parser.add_argument('texture_mean', type=float)
breast_cancer_parser.add_argument('perimeter_mean', type=float)
breast_cancer_parser.add_argument('area_mean', type=float)
breast_cancer_parser.add_argument('smoothness_mean', type=float)
breast_cancer_parser.add_argument('compactness_mean', type=float)
breast_cancer_parser.add_argument('concavity_mean', type=float)
breast_cancer_parser.add_argument('concave_points_mean', type=float)
breast_cancer_parser.add_argument('symmetry_mean', type=float)
breast_cancer_parser.add_argument('radius_se', type=float)
breast_cancer_parser.add_argument('perimeter_se', type=float)
breast_cancer_parser.add_argument('area_se', type=float)
breast_cancer_parser.add_argument('compactness_se', type=float)
breast_cancer_parser.add_argument('concavity_se', type=float)
breast_cancer_parser.add_argument('concave_points_se', type=float)
breast_cancer_parser.add_argument('fractal_dimension_se', type=float)
breast_cancer_parser.add_argument('radius_worst', type=float)
breast_cancer_parser.add_argument('texture_worst', type=float)
breast_cancer_parser.add_argument('perimeter_worst', type=float)
breast_cancer_parser.add_argument('area_worst', type=float)
breast_cancer_parser.add_argument('smoothness_worst', type=float)
breast_cancer_parser.add_argument('compactness_worst', type=float)
breast_cancer_parser.add_argument('concavity_worst', type=float)
breast_cancer_parser.add_argument('concave_points_worst', type=float)
breast_cancer_parser.add_argument('symmetry_worst', type=float)
breast_cancer_parser.add_argument('fractal_dimension_worst', type=float)
breast_cancer_parser.add_argument('frt', type=float)
breast_cancer_parser.add_argument('ftal', type=float)
breast_cancer_parser.add_argument('imensio', type=float)
breast_cancer_parser.add_argument('nsio', type=float)

kidney_parser = reqparse.RequestParser()
kidney_parser.add_argument('Age', type=int)
kidney_parser.add_argument('bp', type=int)
kidney_parser.add_argument('al', type=float)
kidney_parser.add_argument('su', type=float)
kidney_parser.add_argument('rbc', type=float)
kidney_parser.add_argument('pc', type=float)
kidney_parser.add_argument('pcc', type=float)
kidney_parser.add_argument('ba', type=float)
kidney_parser.add_argument('bgr', type=float)
kidney_parser.add_argument('bu', type=float)
kidney_parser.add_argument('sc', type=float)
kidney_parser.add_argument('pot', type=float)

heart_parser = reqparse.RequestParser()
heart_parser.add_argument('Age', type=int)
heart_parser.add_argument('sex', type=int)
heart_parser.add_argument('cp', type=float)
heart_parser.add_argument('trestbps', type=float)
heart_parser.add_argument('chol', type=float)
heart_parser.add_argument('fbs', type=float)
heart_parser.add_argument('restecg', type=float)
heart_parser.add_argument('thalach', type=float)
heart_parser.add_argument('exang', type=float)
heart_parser.add_argument('oldpeak', type=float)
heart_parser.add_argument('slope', type=float)
heart_parser.add_argument('ca', type=float)
heart_parser.add_argument('thal', type=float)

malaria_parser = reqparse.RequestParser()
malaria_parser.add_argument('image', type=FileStorage, location='files')

pneumonia_parser = reqparse.RequestParser()
pneumonia_parser.add_argument('image', type=FileStorage, location='files')

# Resources for prediction endpoints

class DiabetesPrediction(Resource):
    def post(self):
        args = diabetes_parser.parse_args()
        values = [
            args['pregnancies'],
            args['glucose'],
            args['bloodpressure'],
            args['skinthickness'],
            args['insulin'],
            args['bmi'],
            args['dpf'],
            args['age']
        ]
        values = np.asarray(values).reshape(1, -1)
        prediction = diabetes_model.predict(values)
        prediction = str(prediction[0])
        return {'prediction': prediction}

class LiverPrediction(Resource):
    def post(self):
        args = liver_parser.parse_args()
        values = [
            args['Age'],
            args['Gender'],
            args['Total_Bilirubin'],
            args['Direct_Bilirubin'],
            args['Alkaline_Phosphotase'],
            args['Alamine_Aminotransferase'],
            args['Aspartate_Aminotransferase'],
            args['Total_Protiens'],
            args['Albumin'],
            args['Albumin_and_Globulin_Ratio']
        ]
        values = np.asarray(values).reshape(1, -1)
        prediction = liver_model.predict(values)
        prediction = str(prediction[0])
        return {'prediction': prediction}

class BreastCancerPrediction(Resource):
    def post(self):
        args = breast_cancer_parser.parse_args()
        values = [
            args['radius_mean'],
            args['texture_mean'],
            args['perimeter_mean'],
            args['area_mean'],
            args['smoothness_mean'],
            args['compactness_mean'],
            args['concavity_mean'],
            args['concave_points_mean'],
            args['symmetry_mean'],
            args['radius_se'],
            args['perimeter_se'],
            args['area_se'],
            args['compactness_se'],
            args['concavity_se'],
            args['concave_points_se'],
            args['fractal_dimension_se'],
            args['radius_worst'],
            args['texture_worst'],
            args['perimeter_worst'],
            args['area_worst'],
            args['smoothness_worst'],
            args['compactness_worst'],
            args['concavity_worst'],
            args['concave_points_worst'],
            args['symmetry_worst'],
            args['fractal_dimension_worst'],
            args['frt'],
            args['ftal'],
            args['imensio'],
            args['nsio']
        ]
        values = np.asarray(values).reshape(1, -1)
        prediction = breast_cancer_model.predict(values)
        prediction = str(prediction[0])
        return {'prediction': prediction}

class KidneyPrediction(Resource):
    def post(self):
        args = kidney_parser.parse_args()
        values = [
            args['Age'],
            args['bp'],
            args['al'],
            args['su'],
            args['rbc'],
            args['pc'],
            args['pcc'],
            args['ba'],
            args['bgr'],
            args['bu'],
            args['sc'],
            args['pot']
        ]
        values = np.asarray(values).reshape(1, -1)
        prediction = kidney_model.predict(values)
        prediction = str(prediction[0])
        return {'prediction': prediction}

class HeartPrediction(Resource):
    def post(self):
        args = heart_parser.parse_args()
        values = [
            args['Age'],
            args['sex'],
            args['cp'],
            args['trestbps'],
            args['chol'],
            args['fbs'],
            args['restecg'],
            args['thalach'],
            args['exang'],
            args['oldpeak'],
            args['slope'],
            args['ca'],
            args['thal']
        ]
        values = np.asarray(values).reshape(1, -1)
        prediction = heart_model.predict(values)
        prediction = str(prediction[0])
        return {'prediction': prediction}

class MalariaPrediction(Resource):
    def post(self):
        args = malaria_parser.parse_args()
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)
                pred = np.argmax(malaria_model.predict(img)[0])
                return {'prediction': pred}
        except:
            return {'message': 'Please upload an Image'}

class PneumoniaPrediction(Resource):
    def post(self):
        args = pneumonia_parser.parse_args()
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))
                img = img / 255.0
                pred = np.argmax(pneumonia_model.predict(img)[0])
                return {'prediction': pred}
        except:
            return {'message': 'Please upload an Image'}

# Add resources to API
api.add_resource(DiabetesPrediction, '/predict_diabetes')
api.add_resource(LiverPrediction, '/predict_liver')
api.add_resource(BreastCancerPrediction, '/predict_breast_cancer')
api.add_resource(KidneyPrediction, '/predict_kidney')
api.add_resource(HeartPrediction, '/predict_heart')
api.add_resource(MalariaPrediction, '/predict_malaria')
api.add_resource(PneumoniaPrediction, '/predict_pneumonia')

if __name__ == '__app__':
    app.run(debug=True)
