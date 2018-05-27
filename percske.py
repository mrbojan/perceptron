from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
####################################
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
	from sklearn.grid_search import train_test_split
else:
	from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
####################################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
###################################
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(X_train_std,y_train)
y_pred = ppn.predict(X_test_std)
print('Nieprawidłowo sklasyfikowane próbki: %d' %(y_test != y_pred).sum())
###################################
from sklearn.metrics import accuracy_score
print('Dokładnosc: %.2f' % accuracy_score(y_test,y_pred))