# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:08:21 2020

@author: IVAN
"""


import pandas as pd 
dataset = pd.read_csv('avocado.csv')

# Matrix de caracteristicas -> variaveis independentes
Caixa = dataset.iloc[:, 7:8].values
SacosP = dataset.iloc[:, 8:9].values


a = dataset['Total Bags'].corr(dataset['Small Bags'])

# Separar os dados em treinamento
from sklearn.model_selection import train_test_split
Caixa_train, Caixa_test, SacosP_train, SacosP_test = train_test_split(Caixa, SacosP, 
test_size = 0.3, random_state = 0)

## Regressao linerar Simples 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fit -> funcao que ajusta a encontrar a melhor reta o regressor
regressor.fit(Caixa_train, SacosP_train)

# coeficiente: b1 = 0.75146117
regressor.coef_

#Constante : BO =  1949.7466448
regressor.intercept_

# salario = b0 + b1 * experiencia
resultado = 0.7514  + 1949  * 1

### predicao dos resultados do conjunto de teste 

valor_pred = regressor.predict(Caixa_test)

resultado_pred_obs = [valor_pred, SacosP_test]
b = np.concatenate(( np.around(valor_pred,1).reshape(len(valor_pred), 1), SacosP_test.reshape(len(SacosP_test), 1)), 1)

import numpy as np
b= np.reshape([2, 1, 0, 12, 27],(-1,1))
regressor.predict(b)

## visualização de dados
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 14]
plt.scatter(Caixa_train, SacosP_train, color = 'red')
plt.plot(Caixa_train, regressor.predict(Caixa_train), color= 'black')
plt.title('Qtd de  Caixas vs. Sacos Pequenaos')
plt.xlabel('Caixa')
plt.ylabel('Sacolas Pequenas')
plt.show()
