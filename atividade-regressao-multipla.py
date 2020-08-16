# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:22:12 2020

@author: IVAN
"""

# Objetivo prever o lucro da empresa/startup 
# baseado nas variaies depentes: gastos com pd, gastos com adm
# gastos com mkt e o estado. 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import math as mat


#base de dados
dataset = pd.read_csv('hate_crimes.csv', index_col ="state" )
new_dataset = dataset.rename(columns = {"state":"Nome_do_estado", "median_household_income":"Renda_familiar_média",
                          "share_unemployed_seasonal":"população_que_está_desempregada",
                   "share_population_in_metro_areas":"vive_em_áreas_metropolitanas","share_population_with_high_school_degree":"adultos",
                    "share_non_citizen":"nao_cidada", "share_white_poverty":"moradores_brancos_na_pobreza","gini_index":"indice","share_voters_voted_trump":"negros",
                    "hate_crimes_per_100k_splc":"parte_dos_eleitores_presidenciais","avg_hatecrimes_per_100k_fbi":"crimes_de_ódio "}) 

#variaveis
#independente
X = dataset.iloc[:, 1:11].values
#dependente 
# total de crimes
Y = dataset.iloc[:, 0].values


np.any(np.isnan(X))
np.all(np.isfinite(X))
X = X.astype(np.float)
X_train = X.astype(np.float)
X_test = X.astype(np.float)
Y = Y.astype(np.float)
Y_train = Y.astype(np.float)
Y_test = Y.astype(np.float)
#Codificar os dados categoricos: Estado 

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('Date', OneHotEncoder(),[1])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))

X_train = dataset.dropna()
X_test = dataset.dropna()
X = dataset.dropna()
X = dataset.iloc[:, 1:11].values

#excluir a columa que traz redundancia ,
#  problema dummy variable trap
X = X[:, 1:]

#Dividir os dados em conjunto de treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
test_size = 0.3, random_state = 0)

# Aplicar a regressao linear multipla
#  TReinar o nosso modelo 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X = np.array(X)
Y = np.array(Y)
regressor.fit(X_train, Y_train)

# previsao dos dados de teste , como o modelo se comporta
y_pred = regressor.predict(X_test)
resultado_pred_obs = [y_pred, Y_test]
b = np.concatenate(( np.around(y_pred,1).reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1)
compara_ys = np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test), 1)),1)

print(regressor.intercept_)
print(regressor.coef_)

dataset.corr()


## BackWard Elimination ##
from sklearn.metrics import mean_squared_error
from math import sqrt

mse = round((mean_squared_error(Y_test, y_pred))/100, 3)
rmse = round((sqrt(mse))/100, 3)

## passo 1 : Selecionar o nivel de significancaica (ns= 0.05)
## passo 2: Construir o modelo 
## passo 3 : Considera o preditor  com p-value mais alto.
# Se p > NS vou para o passo 4 , senao termina
##  passo 4 : retirar o preditor com p-value > NS = 0.05
## passo 5 : constriur o modelo sem esse preitor

dataset = pd.get_dummies(dataset)
x = dataset.drop(['median_household_income'], axis = 1)
y = dataset['median_household_income']

#verificar os valores 
import statsmodels.api as sm
x = sm.add_constant(x )

x = dataset.iloc[:, 1:11].values
x = dataset.dropna()
modelo = sm.OLS(Y, x).fit()
modelo.summary()

# valores que nao tem impacto com lucro tem que retirar do modelo
# p > NS
# x7 = 0.657 > 0.05
X1 = x.drop(['share_non_white'], axis = 1 )
modelo = sm.OLS(Y, X1).fit()
modelo.summary()

x =  dataset.rename(columns = {"0":"Nome_do_estado", "1":"Renda_familiar_média",
                          "2":"população_que_está_desempregada",
                   "3":"vive_em_áreas_metropolitanas","4":"adultos",
                    "5":"nao_cidada", "6":"moradores_brancos_na_pobreza","7":"indice","8":"negros",
                    "9":"parte_dos_eleitores_presidenciais"}) 
## p > NS
# mkt = 0.06 > 0.05
X2 = X1.drop(['share_population_in_metro_areas'], axis = 1)
modelo = sm.OLS(Y, X2).fit()
modelo.summary()

X3 = X2.drop(['avg_hatecrimes_per_100k_fbi'], axis = 1)
modelo = sm.OLS(Y, X3).fit()
modelo.summary()

X4 = X3.drop(['gini_index'], axis = 1)
modelo = sm.OLS(Y, X4).fit()
modelo.summary()

X5 = X4.drop(['hate_crimes_per_100k_splc'], axis = 1)
modelo = sm.OLS(Y, X5).fit()
modelo.summary()

X6 = X5.drop(['share_non_citizen'], axis = 1)
modelo = sm.OLS(Y, X6).fit()
modelo.summary()

X7 = X6.drop(['share_unemployed_seasonal'], axis = 1)
modelo = sm.OLS(Y, X7).fit()
modelo.summary()



X_train, X_test, Y_train, Y_test = train_test_split(X7, Y, 
test_size = 0.3, random_state = 0)

# Aplicar a regressao linear multipla
#  TReinar o nosso modelo 

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred2 = regressor.predict(X_test)
compara_ys = np.concatenate((y_pred2.reshape(len(y_pred),1), Y_test.reshape(len(Y_test), 1)),1)

mse2 = round((mean_squared_error(Y_test, y_pred2))/100, 2)
rmse2 = round((sqrt(mse2))/100, 2)

print({"todas_as_variaveis": {'mse': mse, 'rmse': rmse }})
print({"backWard_elimination": {'mse2': mse2, 'rmse2': rmse2 }})