# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:08:21 2020

@author: IVAN
"""


import pandas as pd 
dataset = pd.read_csv('avocado.csv')

# Matrix de caracteristicas -> variaveis independentes
varpreco = dataset.iloc[:, 2:3].values
vendas = dataset.iloc[:, 3:4].values
vartest = dataset.iloc[:, 2:4].values

a = dataset['AveragePrice'].corr(dataset['Total Volume'])


# Separar os dados em treinamento
from sklearn.model_selection import train_test_split
varpreco_train, varpreco_test, vendas_train, vendas_test = train_test_split(varpreco, vendas, 
test_size = 0.3, random_state = 0)

## Regressao linerar Simples 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# fit -> funcao que ajusta a encontrar a melhor reta o regressor
regressor.fit(varpreco_train, vendas_train)

# coeficiente: b1 = -1589056.18180111
regressor.coef_

#Constante : BO =  3049982.2618458224
regressor.intercept_

b = np.concatenate(( np.around(preco_pred,1).reshape(len(preco_pred), 1), preco_test.reshape(len(preco_test), 1)), 1)

# salario = b0 + b1 * qtd vendas
resultado =  3049982.2618 + -1589056.1818 * 2

### predicao dos resultados do conjunto de teste 
## variação no total de vendas de cada ano
valor_vendas = regressor.predict(varpreco_test)

resultado_pred_obs = [valor_vendas, vendas_test]

import numpy as np
valorvendaQTD = np.reshape([2, 1, 0, 12, 27],(-1,1))
regressor.predict(valorvendaQTD)

## visualização de dados
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 12]
plt.scatter(varpreco_train, vendas_train, color = 'red')
plt.plot(varpreco_train, regressor.predict(varpreco_train), color= 'black')
plt.title('Variação do preço vs. Vendas')
plt.xlabel('VAriação de preçõ determinando a quantidade')
plt.ylabel('Total de Vendas')
plt.show()
