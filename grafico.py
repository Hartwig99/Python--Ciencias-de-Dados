# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:27:04 2020

@author: IVAN
"""


##*** funcao drafico
import matplotlib.pyplot as plt

def histograma_binds(d):
 	for i,largura_bin in enumerate([1, 5, 10, 15])

# cria plot 
ax = plt.subplot(2,2, i+1)

# desenha o grafico 
ax.hist(d, bins = int(180/largura_bin), color = 'blue', edgecolor= 'black')

# titulos e labels 
ax.set_title('Bin = %d' % largura_bin, size = 30)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Frequencia')
