# -*- coding: utf-8 -*-
import pandas as pd
import pickle

from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# Load model
model = pickle.load(open('model/rossmann_model.pkl', 'rb'))

# Initialize API
app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json:# Request returned data
        if isinstance(test_json, dict):# Single json
            test_raw = pd.DataFrame(test_json, index = [0])
        else:# Multiple json
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
        
        # Instantiate Rossmann class
        pipeline = Rossmann()
        
        # Data cleaning
        df1 = pipeline.clean_data(test_raw)
        
        # Feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # Data preparation
        df3 = pipeline.data_preparation(df2)
        
        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
    
    else:# Request returned no data
        return Response('{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    from os import environ
    app.run(host = '0.0.0.0', port = environ.get('PORT', 5000))
