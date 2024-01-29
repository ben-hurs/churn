from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

app = Flask(__name__)

# Carregar o modelo
modelo = joblib.load('logistic_model.joblib')

# Colunas utilizadas no treinamento do modelo
colunas_treinamento = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# Rota para fazer previsões
@app.route('/prever', methods=['POST'])
def prever():
    dados = request.get_json()

    # Converter os dados de entrada em um DataFrame
    df = pd.DataFrame([dados])

    # Garantir que as colunas estejam na mesma ordem que no treinamento do modelo
    df = df[colunas_treinamento]

    # Lidar com variáveis categóricas
    categoricas = df.select_dtypes(include='object')
    numericas = df.select_dtypes(exclude='object')

    # Codificar variáveis categóricas usando OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    categoricas_encoded = pd.DataFrame(encoder.fit_transform(categoricas), columns=encoder.get_feature_names_out(categoricas.columns))

    # Concatenar variáveis codificadas com variáveis numéricas
    df_encoded = pd.concat([categoricas_encoded, numericas], axis=1)

    # Normalizar dados
    scaler = MinMaxScaler()
    dados_array = scaler.fit_transform(df_encoded)

    # Fazer a previsão
    previsao = modelo.predict(dados_array.reshape(1, -1))

    # Resultado em JSON
    resultado = {'previsao': int(previsao[0])}

    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
