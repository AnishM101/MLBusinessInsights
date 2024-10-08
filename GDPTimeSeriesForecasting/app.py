import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Blueprint, request, jsonify, render_template, send_from_directory, redirect, url_for

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

        def generate_features(df):
            lag1 = df['GDP (Billion USD)'].iloc[-1]
            lag2 = df['GDP (Billion USD)'].iloc[-2]
            rolling_mean = df['GDP (Billion USD)'].rolling(window = 4).mean().iloc[-1]
            rolling_std = df['GDP (Billion USD)'].rolling(window = 4).std().iloc[-1]
            rolling_min = df['GDP (Billion USD)'].rolling(window = 4).min().iloc[-1]
            rolling_max = df['GDP (Billion USD)'].rolling(window = 4).max().iloc[-1]
            gdp_quarter_diff = lag1 - lag2
            gdp_year_diff = df['GDP (Billion USD)'].iloc[-1] - df['GDP (Billion USD)'].iloc[-4]
            quarter_sin = np.sin(2 * np.pi * quarter / 4)
            quarter_cos = np.cos(2 * np.pi * quarter / 4)

            return {
                'Year': [year],
                'Quarter': [quarter],
                'Lag1': [lag1],
                'Lag2': [lag2],
                'RollingMean': [rolling_mean],
                'RollingStd': [rolling_std],
                'RollingMin': [rolling_min],
                'RollingMax': [rolling_max],
                'GDPQuarterDiff': [gdp_quarter_diff],
                'GDPYearDiff': [gdp_year_diff],
                'QuarterSin': [quarter_sin],
                'QuarterCos': [quarter_cos]
            }

        input_data = pd.DataFrame(generate_features(df))
        preprocessed_features = model.named_steps['preprocessor'].transform(input_data)
        prediction = model.named_steps['ridge'].predict(preprocessed_features)[0]

        plt.figure(figsize = (10, 6))

        current_year = df['Year'].max()

        selected_year_quarter = f'{year}-Q{quarter}'

        if year <= current_year:
            actual_segment = df[df['Year-Quarter'] <= selected_year_quarter]
            sns.lineplot(x = actual_segment['Year-Quarter'], y = actual_segment['GDP (Billion USD)'], label = 'Actual GDP', marker = 'o', color = 'blue')
        else:
            actual_segment = df[df['Year-Quarter'] <= f'{current_year}-Q4']
            sns.lineplot(x = actual_segment['Year-Quarter'], y = actual_segment['GDP (Billion USD)'], label = 'Actual GDP', marker = 'o', color = 'blue')

            future_segment = df[df['Year-Quarter'] > f'{current_year}-Q4']
            sns.lineplot(x = future_segment['Year-Quarter'], y = future_segment['GDP (Billion USD)'], label = 'Forecasted GDP', linestyle = '--', color = 'orange')

            plt.scatter([selected_year_quarter], [prediction], color = 'green', label = 'Predicted GDP', s = 100)

        start_year = max(year - 5, df['Year'].min())
        end_year = year + 5

        filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
        
        filtered_quarters = filtered_df['Year-Quarter']
        plt.xlim(filtered_df['Year-Quarter'].iloc[0], filtered_df['Year-Quarter'].iloc[-1])

        plt.xticks(rotation = 45)

        plt.title(f'GDP Prediction for {year}-Q{quarter}')
        plt.xlabel('Year-Quarter')
        plt.ylabel('GDP (Billion USD)')
        plt.xticks(rotation = 45)
        plt.legend()

        plot_filename = 'forecast.png'
        plot_filepath = os.path.join('GDPTimeSeriesForecasting', plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

        return redirect(url_for('time_app.result', year = year, quarter = quarter, plot_filename = plot_filename))

    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

@time_app.route('/result/<int:year>/<int:quarter>', methods = ['GET'])
def result(year, quarter):
    plot_filename = request.args.get('plot_filename', '')
    plot_filepath = os.path.join('GDPTimeSeriesForecasting', plot_filename)

    if not os.path.exists(plot_filepath):
        return jsonify({'error': 'Plot not found'}), 404
    
    return render_template('time_result.html', year = year, quarter = quarter, plot_filename = plot_filename)

@time_app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'GDPTimeSeriesForecasting'), filename)

@time_app.route('/time_form.css')
def form_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'GDPTimeSeriesForecasting'), 'time_form.css')

@time_app.route('/time_result.css')
def result_styles():
    return send_from_directory(os.path.join(os.getcwd(), 'GDPTimeSeriesForecasting'), 'time_result.css')