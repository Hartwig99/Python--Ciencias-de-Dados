# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:48:01 2020

@author: IVAN
"""


import numpy as np 
import pandas as pd 


dataset = pd.read_csv('hotel_bookings.csv')


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
dados_perdidos = SimpleImputer(missing_values = np.nan, strategy='mean', verbose=0)

dados_perdidos = dados_perdidos.fit(X[:,1:3 ])
X[:,1:3 ] = dados_perdidos.transform(X[:,1:3 ])

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OrdinalEncoder 


code_X = ColumnTransformer([('codificar_X', OrdinalEncoder(), [0])], remainder='passthrough')
a = np.array(code_X.fit_transform(X))