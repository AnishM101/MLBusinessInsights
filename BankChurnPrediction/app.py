import pickle
import os
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, send_from_directory, redirect, url_for

churn_app = Blueprint('churn_app', __name__, template_folder = '.', static_folder = '.')

with open('BankChurnPrediction/model.pkl', 'rb') as f:
    model_data = pickle.load(f)

if not model_data:
    raise Exception('Model not loaded correctly')

model = model_data['model']
threshold = model_data['threshold']

@churn_app.route('/')
def home():
    return render_template('churn_form.html')

@churn_app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.form
        required_fields = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        
        for field in required_fields:
            if field not in data or data[field] == '':
                raise ValueError(f'Missing or empty field: {field}')
        
        features = {
            'CreditScore': [int(data['CreditScore'])],
            'Gender': [int(data['Gender'])],
            'Age': [int(data['Age'])],
            'Tenure': [int(data['Tenure'])],
            'Balance': [float(data['Balance'])],
            'NumOfProducts': [int(data['NumOfProducts'])],
            'HasCrCard': [int(data['HasCrCard'])],
            'IsActiveMember': [int(data['IsActiveMember'])],
            'EstimatedSalary': [float(data['EstimatedSalary'])]
        }
        
        features_df = pd.DataFrame(features)

        if features_df.isnull().values.any():
            raise ValueError('Dataframe contains NaN values')

        features_df['BalanceSalaryRatio'] = features_df['Balance'] / features_df['EstimatedSalary']
        features_df['TenureByAge'] = features_df['Tenure'] / features_df['Age']
        features_df['CreditScorePerAge'] = features_df['CreditScore'] / features_df['Age']
        
        predictions_prob = model.predict_proba(features_df)[:, 1]
        prediction = (predictions_prob >= threshold).astype(int)
        
        return redirect(url_for('churn_app.result', prediction = int(prediction[0])))
    
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

@churn_app.route('/result/<int:prediction>', methods = ['GET'])
def result(prediction):
    return render_template('churn_result.html', prediction = prediction)

@churn_app.route('/churn_form.css')
def form_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'BankChurnPrediction'), 'churn_form.css')

@churn_app.route('/churn_result.css')
def result_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'BankChurnPrediction'), 'churn_result.css')