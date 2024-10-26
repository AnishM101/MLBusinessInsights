import os
import pickle
from flask import Flask, render_template, send_from_directory, request, jsonify
from BankChurnPrediction.app import churn_app
from GDPTimeSeriesForecasting.app import time_app
from GroceryStoreSalesPrediction.app import sales_app

app = Flask(__name__, template_folder = '.')

app.register_blueprint(churn_app, url_prefix = '/churn')
app.register_blueprint(time_app, url_prefix = '/time')
app.register_blueprint(sales_app, url_prefix = '/sales')

with open('ChatBot/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/styles.css')
def home_styles():
    return send_from_directory('.', 'styles.css')

@app.route('/chat', methods = ['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"response": "Sorry, I didn't get that. Could you rephrase?"})
    
    predicted_class_index = model.predict([user_input])[0]
    response = model.named_steps['labelencoder'].inverse_transform([predicted_class_index])[0]

    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port, debug = True)