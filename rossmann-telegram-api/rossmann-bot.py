# -*- coding: utf-8 -*-
import json
import pandas as pd
import requests

from flask import Flask, request, Response

# Telegram bot token
token = '2138030119:AAEUuVTNb858e_3Kyf01enJ9O7GsL8mYeBo'

# Webhook
# https://api.telegram.org/bot2138030119:AAEUuVTNb858e_3Kyf01enJ9O7GsL8mYeBo/setWebhook?url=https://rossmannbot-telegram.herokuapp.com/
# https://api.telegram.org/bot2138030119:AAEUuVTNb858e_3Kyf01enJ9O7GsL8mYeBo/deleteWebhook

def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/sendMessage'.format(token)
    url = url+'?chat_id={}'.format(chat_id)
    
    r = requests.post(url, json = {'text': text})
    print('Status code {}'.format(r.status_code))

def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']
    store_id = store_id.replace('/', '')
    try:
        store_id = int(store_id)
    except:
        store_id = None
    return chat_id, store_id

def load_dataset(store_id):
    # Load store info
    df_store_raw = pd.read_csv('store.csv')
    
    # Load test dataset
    df10 = pd.read_csv('test.csv')
    df_test = pd.merge(df10, df_store_raw, how = 'left', on = 'Store')
    
    # Choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]
    
    if df_test.empty:
        return None
    else:
        return df_test

def clean_data(df_test):
    # Remove days when store is closed
    df_test = df_test[df_test['Open'] != 0]
    df_test = df_test[~df_test['Open'].isnull()]
    df_test = df_test.drop('Id', axis = 1)
    
    # Convert dataframe to json
    data = json.dumps(df_test.to_dict(orient = 'records'))
    
    return data

def predict(data):
    # API call
    url = 'https://rossmann-store-sales.herokuapp.com/predict'
    header = {'Content-type': 'application/json'}
    data = data
    
    r = requests.post(url, data = data, headers = header)
    print('Status code {}'.format(r.status_code))
    
    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())
    
    return d1

# Initialize API
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.get_json()
        chat_id, store_id = parse_message(message)
        if store_id is None:
            send_message(chat_id, 'Invalid store id')
            return Response('Ok', status=200)
        else:
            # Load/clean data
            df = load_dataset(store_id)
            if df is None:
                send_message(chat_id, 'Store not available')
                return Response('Ok', status=200)
            else:
                data = clean_data(df)
                
                # Prediction
                d1 = predict(data)
                
                # Calculation
                d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()
                
                # Send message
                msg = 'Store {} will sell $ {:,.2f} in the next 6 weeks'.format(
                    d2.loc[0, 'store'], 
                    d2.loc[0, 'prediction'])
                send_message(chat_id, msg)
                return Response('Ok', status=200)
    else:
        return '<h1>Rossmann Telegram BOT</h1>'

if __name__ == '__main__':
    from os import environ
    app.run(host = '0.0.0.0', port = environ.get('PORT', 5000))
