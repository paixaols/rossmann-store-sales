# -*- coding: utf-8 -*-
import datetime
import inflection
import math
import numpy as np
import os
import pandas as pd
import pickle

class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        self.competition_distance_scaler = pickle.load(open(os.path.join(self.home_path, 'parameters/competition_distance_scaler.pkl'), 'rb'))
        self.competition_time_month_scaler = pickle.load(open(os.path.join(self.home_path, 'parameters/competition_time_month_scaler.pkl'), 'rb'))
        self.promo_time_week_scaler = pickle.load(open(os.path.join(self.home_path, 'parameters/promo_time_week_scaler.pkl'), 'rb'))
        self.year_scaler = pickle.load(open(os.path.join(self.home_path, 'parameters/year_scaler.pkl'), 'rb'))
        self.store_type_scaler = pickle.load(open(os.path.join(self.home_path, 'parameters/store_type_scaler.pkl'), 'rb'))
    
    def clean_data(self, df1):
        ## 1.1 Renomear colunas
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']
        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        df1.columns = cols_new
        
        ## 1.3 Tipos dos dados
        df1['date'] = pd.to_datetime(df1['date'])
        
        # Competition distance | replace with distance larger than previous max
        df1['competition_distance'].fillna(200000, inplace = True)
        
        # Competition open since month | replace with moth in date column
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis = 1)
        
        # Competition open since year | replace with year in date column
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis = 1)
        
        # Promo 2 since week | replace with week in date column
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis = 1)
        
        # Promo 2 since year | replace with year in date column
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis = 1)
        
        # Promo interval
        # Get month of date column in str format
        month_str = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df1['month_str'] = df1['date'].apply(lambda x: month_str[x.month])
        
        # 1: promo active, 0: otherwise
        df1['promo_interval'].fillna('-', inplace = True)
        df1['promo2_active'] = df1.apply(lambda x: 1 if x['month_str'] in x['promo_interval'].split(',') else 0, axis = 1)
        
        ## 1.5 Tipos dos dados
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
        df1['promo2_since_week'] = df1['promo2_since_week'].astype('int64')
        df1['promo2_since_year'] = df1['promo2_since_year'].astype('int64')
        
        return df1
    
    def feature_engineering(self, df2):
        ## 2.4 Criação das variáveis
        # Ano
        df2['year'] = df2['date'].apply(lambda x: x.year)
        
        # Mês
        df2['month'] = df2['date'].apply(lambda x: x.month)
        
        # Dia
        df2['day'] = df2['date'].apply(lambda x: x.day)
        
        # Semana
        df2['week_of_year'] = df2['date'].apply(lambda x: x.weekofyear)
        
        # Data | ano-semana
        df2['year_week'] = df2['date'].apply(lambda x: x.strftime('%Y-%W'))
        
        # Competition since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year = x['competition_open_since_year'], month = x['competition_open_since_month'], day = 1), axis = 1)
        df2['competition_time_month'] = ((df2['date']-df2['competition_since'])).apply(lambda x: x.days/30).astype('int64')
        
        # Promo since
        df2['promo2_since'] = df2['promo2_since_year'].astype('str')+'-'+df2['promo2_since_week'].astype('str')
        df2['promo2_since'] = df2['promo2_since'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%W'))
        df2['promo_time_week'] = (df2['date']-df2['promo2_since']).apply(lambda x: x.days/7).astype('int64')
        
        # Assortment | a = basic, b = extra, c = extended
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        
        # State holiday | a = public holiday, b = Easter holiday, c = Christmas, 0 = None
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public holiday' if x == 'a' else 'easter holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular day')
        
        # 3 Filtragem de variáveis
        ## 3.1 Filtragem das linhas
        df2 = df2[df2['open'] == 1]
        
        ## 3.2 Seleção das colunas
        drop_cols = ['open', 'promo_interval', 'month_str']
        df2.drop(drop_cols, axis = 1, inplace = True)
        
        return df2
    
    def data_preparation(self, df5):
        ## 5.2 Rescaling
        # Competition distance
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']].values)
        
        # Competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_time_month']].values)
        
        # Promo time week
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5[['promo_time_week']].values)
        
        # Year
        df5['year'] = self.year_scaler.transform(df5[['year']].values)
        
        ### 5.3.1 Encoding
        # state_holiday - one hot encoding
        df5 = pd.get_dummies(df5, prefix = ['state_holiday'], columns = ['state_holiday'])
        
        # store_type - label encoding
        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])
        
        # assortment - ordinal encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)
        
        ### 5.3.3 Transformação de natureza
        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*2*np.pi/7))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*2*np.pi/7))

        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*2*np.pi/30))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*2*np.pi/30))

        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x*2*np.pi/52))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x*2*np.pi/52))

        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*2*np.pi/12))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*2*np.pi/12))
        
        features_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 
                             'competition_open_since_month', 'competition_open_since_year', 'promo2', 
                             'promo2_since_week', 'promo2_since_year', 'competition_time_month', 
                             'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'day_sin', 'day_cos', 
                             'week_of_year_sin', 'week_of_year_cos', 'month_sin', 'month_cos']
        return df5[features_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # Prediction
        pred = model.predict(test_data)
        
        # Join original data and the prediction
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient = 'records', date_format = 'iso')
