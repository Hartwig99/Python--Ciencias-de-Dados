import numpy as np 
import pandas as pd 


dataset = pd.read_cvsv('dados.csv')


X = dataset.iloc[:, :-1].values

Y = dataset.iloc[:, 3].values

from sklearn.inpute import SimpleInputer
dados_perdidos = SimpleInputer(missing.values = np.man, strategy='mean', verbose=0)

dados_perdidos = dados_perdidos.fiti(X[:,1:3 ])
X[:,1:3 ] = dados_perdidos.transform(X[:,1:3 ])