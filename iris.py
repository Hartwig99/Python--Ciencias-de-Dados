# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:26:12 2020

@author: IVAN
"""
import pandas as pd 


# pre visualização de dados 

iris = pd.read_csv("iris.csv", sep=', ')



# checar tamanho 
iris.shape 

#checar nome das colunas 
iris.columns

#olhando as primeiras 10 linha 
iris.iloc[0:10,:]

iris.head()
iris.tail()

#obter uma amostra randomica 
iris.sample(5)


#recuperando 10 primeiros dados dde uma coluna 
iris['"sepal.length"'][0:10]

# explorar variaveis individuais ## 
resumo = iris['sepal.length'].describe()


# media, mediana, desvio padrao, variancia  
iris.mean()
iris.median()
iris.std()
iris.var()


## graficos de pre visualização ##

## Histograma ## 

import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] = [12,12]

d = iris['sepal.lenght']

#cria o histograma
plt.hist(x=d, bins='auto', color='#0523bb' )
plt.xlabel('Sepal Length')
plt.ylabel('Frequencia')
plt.title('Histograma da variavel')

#histograma_bins(d)










