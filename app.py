import os
from flask import Flask, render_template, send_from_directory, redirect, url_for
from BankChurnPrediction.app import churn_app
from GDPTimeSeriesForecasting.app import time_app
from GroceryStoreSalesPrediction.app import sales_app

app = Flask(__name__, template_folder = '.')

app.register_blueprint(churn_app, url_prefix = '/churn')
app.register_blueprint(time_app, url_prefix = '/time')
app.register_blueprint(sales_app, url_prefix = '/sales')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/styles.css')
def home_styles():
    return send_from_directory('.', 'styles.css')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port, debug = True)