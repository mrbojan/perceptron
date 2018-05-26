import pandas as pd
from perceptron import Perceptron 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
df.tail()

import matplotlib.pyplot as plt
import numpy as np
print(df.tail())

y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values

ppn = Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.figure(1)
plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_,marker='o')
plt.xlabel('Epoko')
plt.ylabel('Liczba aktualizacji')
plt.figure(2)
plt.scatter(X[:50,0], X[:50,1],color='red',marker='o',label='Setosa')
plt.scatter(X[50:100,0], X[50:100,1],color='blue',marker='x',label='Versicolor')
plt.xlabel('Dlugosc dzialko [cm]')
plt.ylabel('Dlugosc platka [cm]')
plt.legend(loc='upper left')
plt.show()