import pickle
import os
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, send_from_directory

sales_app = Blueprint('sales_app', __name__, template_folder = '.', static_folder = '.')

with open('GroceryStoreSalesPrediction/model.pkl', 'rb') as f:
    model = pickle.load(f)

if not model:
    raise Exception('Model not loaded correctly')

@sales_app.route('/')
def home():
    return render_template('sales_form.html')

@sales_app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.form
        required_fields = ['ItemWeight', 'ItemFatContent', 'ItemVisibility', 'ItemType', 'ItemMRP', 'OutletSize', 'YearsSinceEstablishment']

        for field in required_fields:
            if field not in data or data[field] == '':
                raise ValueError(f'Missing or empty field: {field}')
        
        features = {
            'ItemWeight': [float(data['ItemWeight'])],
            'ItemFatContent': [int(data['ItemFatContent'])],
            'ItemVisibility': [float(data['ItemVisibility'])],
            'ItemType': [int(data['ItemType'])],
            'ItemMRP': [float(data['ItemMRP'])],
            'OutletSize': [int(data['OutletSize'])],
            'YearsSinceEstablishment': [int(data['YearsSinceEstablishment'])]
        }

        features_df = pd.DataFrame(features)

        if features_df.isnull().values.any():
            raise ValueError('Dataframe contains NaN values')

        preprocessed_features = model.named_steps['preprocessor'].transform(features_df)
        prediction = model.named_steps['ensemble'].predict(preprocessed_features)
        
        rounded_prediction = round(float(prediction[0]), 2)

        return render_template('sales_result.html', prediction = rounded_prediction)
    
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

@sales_app.route('/sales_form.css')
def form_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'GroceryStoreSalesPrediction'), 'sales_form.css')

@sales_app.route('/sales_result.css')
def result_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'GroceryStoreSalesPrediction'), 'sales_result.css')