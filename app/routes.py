# app/routes.py
from flask import render_template, request
from app import app  # Importa a instância do Flask diretamente
import joblib
import pandas as pd

# Carregar o modelo
model = joblib.load('logistic_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obter dados do formulário
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        # Criar um DataFrame com os dados do usuário
        user_data = pd.DataFrame([input_data])

        # Realizar a previsão usando o modelo
        prediction = model.predict(user_data)[0]

        return render_template('result.html', prediction=prediction)
