import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import Algorithms

#import dataset e studio correlazione
df = pd.read_csv("https://raw.githubusercontent.com/Cionsa/Datasets/main/hour.csv", delimiter=',')
data = df.drop(['instant', 'registered', 'casual', 'dteday'], axis=1)

Algorithms.heatmap(data)

X,y = Algorithms.scaledata(data)

#split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#random forest
Algorithms.rf(X_train, X_test, y_train, y_test)

#decision tree
Algorithms.dt(X_train, X_test, y_train, y_test)

#AdaBoost
Algorithms.ada(X_train, X_test, y_train, y_test)

#GradientBoostingTree
Algorithms.gbt(X, y, X_train, X_test, y_train, y_test)