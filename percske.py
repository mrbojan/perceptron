from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
####################################
from distutils.version import LooseVersion as Version
from sklearn import _version_ as sklearn_version
if Version(sklearn_version) < '0.18':
	from sklearn.grid_search import train_test_split
else:
	from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)