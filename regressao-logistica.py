# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:46:23 2020

@author: IVAN
"""


#importando as bilbiotecas 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# importando a base de dados
dataset = pd.read_csv('compra.csv')
# Correlação entre Inteligencia e combate
#a = dataset['Intelligence'].corr(dataset['Combat'])

#seleciona variavel inteligencia e combate
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#treinamento regressão logistica

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(random_state=0)
classificador.fit(X_train, y_train)

## predizer novo resultado 
idade = 30
salario = 80000
classificador.predict(sc.transform ([[idade, salario]]))
classificador.predict_proba(sc.transform ([[idade, salario]]))

y_pred = classificador.predict(X_test)
print(np.concatenate(y_pred.reshape(len(y_test.reshape(len(y_test),1)),1)))


## Matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import plot_confusion_matrix 
plt.rcParams['figure.figsize'] = [12, 12]
plot_confusion_matrix(classificador, X_test, y_test)
plot_confusion_matrix(classificador, X_test, y_test, normalize = True)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
