# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:34:12 2020

@author: IVAN
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:49:07 2020

@author: IVAN
"""


#importando as bilbiotecas 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# importando a base de dados
dataset = pd.read_csv('poderes.csv')
# Correlação entre Inteligencia e combate
a = dataset['Intelligence'].corr(dataset['Combat'])

#seleciona variavel inteligencia e combate
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 6:7].values


# treinamento do modelo de regressao linear simples
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Treinamento com  o modelo de regressão polinomial
from sklearn.preprocessing import PolynomialFeatures
reg_poli = PolynomialFeatures(degree=2)

X_poli = reg_poli.fit_transform(X)

# y = b0 + b1 * x1 + b2 * x1^2 + b3 * x1^3
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poli, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Regressão Linear Simples')
plt.xlabel('Inteligencia')
plt.ylabel('Combate')
plt.show()

# Visualizando os resultados da Regressão Polinomial
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'green')
plt.plot(X, lin_reg2.predict(X_poli), color = 'blue')
plt.title('Regressão Linear Polinomial')
plt.xlabel('Inteligencia')
plt.ylabel('Combate')
plt.show()

# valor predito com regresão simples
lin_reg.predict([[30]])

# valor predito com reg polinomial
lin_reg2.predict(reg_poli.fit_transform([[30]]))

# metricas de qualidade
# RLS
from sklearn.metrics import mean_squared_error
mse_rls = mean_squared_error(y, lin_reg.predict(X))
rmse_rls = round(np.sqrt(mse_rls),2)

res_rls = {"MSE_RLS": mse_rls, "RMSE_RLS": rmse_rls}
print(res_rls)

# RP
mse_rp = mean_squared_error(y, lin_reg2.predict(X_poli))
rmse_rp = round(np.sqrt(mse_rp),2)

res_rp = {"MSE_RP": mse_rp, "RMSE_RP": rmse_rp}
print(res_rp)


##### SVR #####
## transformando em matriz
y_mat = y.reshape(len(y),1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_sc = sc_X.fit_transform(X)
y_sc = sc_y.fit_transform(y_mat)

# Treinamento na base de dados toda
# # https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200
from sklearn.svm import SVR
reg_svr = SVR(kernel='rbf')
reg_svr.fit(X_sc, y_sc)

sc_y.inverse_transform(reg_svr.predict(sc_X.transform([[30]])))

# SVR
mse_svr = mean_squared_error(y_sc, sc_y.inverse_transform(reg_svr.predict(X_sc)))
rmse_svr = round(np.sqrt(mse_svr),2)

res_svr = {"MSE_SVR": mse_svr, "RMSE_SVR": rmse_svr}
print(res_svr)

# Visualizacao
plt.scatter(sc_X.inverse_transform(X_sc), sc_y.inverse_transform(y_sc), color = 'green')
plt.plot(sc_X.inverse_transform(X_sc), sc_y.inverse_transform(reg_svr.predict(X_sc)), color = 'blue')
plt.title('SVR')
plt.xlabel('Inteligencia')
plt.ylabel('Combate')
plt.show()