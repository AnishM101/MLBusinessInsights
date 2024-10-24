import pickle
import os
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, send_from_directory

time_app = Blueprint('time_app', __name__, template_folder = '.', static_folder = '.')

with open('GDPTimeSeriesForecasting/model.pkl', 'rb') as f:
    model = pickle.load(f)

if not model:
    raise Exception('Model not loaded correctly')

@time_app.route('/')
def home():
    return render_template('time_form.html')

@time_app.route('/predict', methods = ['POST'])
def predict():
    try:
        year = int(request.form['year1'] + request.form['year2'] + request.form['year3'] + request.form['year4'])
        quarter = int(request.form['quarter'])
        
        data_file = os.path.join(os.getcwd(), 'GDPTimeSeriesForecasting', 'data.csv')

        df = pd.read_csv(data_file)
        df['Year'] = df['Year-Quarter'].str[:4].astype(int)
        df['Quarter'] = df['Year-Quarter'].str[-1].astype(int)

        selected_year_quarter = f'{year}-Q{quarter}'
        existing_record = df[df['Year-Quarter'] == selected_year_quarter]

        if not existing_record.empty:
            actual_gdp = existing_record['GDP (Billion USD)'].values[0]
            formatted_gdp = round(float(actual_gdp), 2)
            return render_template('time_result.html', year=year, quarter=quarter, gdp=formatted_gdp, type="Actual")
        else:
            def generate_features(df):
                rolling_mean = df['GDP (Billion USD)'].rolling(window = 4).mean().iloc[-1]
                rolling_std = df['GDP (Billion USD)'].rolling(window = 4).std().iloc[-1]
                rolling_min = df['GDP (Billion USD)'].rolling(window = 4).min().iloc[-1]
                rolling_max = df['GDP (Billion USD)'].rolling(window = 4).max().iloc[-1]

                return {
                    'Year': [year],
                    'Quarter': [quarter],
                    'RollingMean': [rolling_mean],
                    'RollingStd': [rolling_std],
                    'RollingMin': [rolling_min],
                    'RollingMax': [rolling_max]
                }

            input_data = pd.DataFrame(generate_features(df))
            preprocessed_features = model.named_steps['preprocessor'].transform(input_data)
            forecast = model.named_steps['ridge'].predict(preprocessed_features)[0]
            formatted_gdp = round(float(forecast), 2)

            return render_template('time_result.html', year = year, quarter = quarter, gdp = formatted_gdp, type = "Forecast")

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

@time_app.route('/time_form.css')
def form_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'GDPTimeSeriesForecasting'), 'time_form.css')

@time_app.route('/time_result.css')
def result_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'GDPTimeSeriesForecasting'), 'time_result.css')