from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os

modelo = pickle.load(open('../../models/modelo.sav','rb'))

colunas = ['tamanho', 'ano', 'garagem']
# df = pd.read_csv(r"./MLOps/casas.csv")
# # df = df[colunas]
# X = df.drop(columns=['preco'])
# y = df['preco']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# from sklearn.linear_model import LinearRegression
# modelo = LinearRegression()
# modelo.fit(X_train,y_train)

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME') #andre
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD') #alura

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    #frase = "Python é ótimo para Machine Learning"
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang="pt-br",to="en")
    polaridade = tb_en.sentiment.polarity
    return f"polaridade: {polaridade}"

@app.route('/cotacao/',methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    print(dados_input)
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True, host='0.0.0.0')