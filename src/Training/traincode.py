import pandas as pd
from sklearn.model_selection import train_test_split
import Training.Algorithms
import dbConnection.dbConnection

def traincode():
    engine, gatherings_detection, gatherings_prediction, connection = dbConnection.dbConnection.connect()
    #data, prediction_df = dbConnection.dbConnection.getTables(connection)
    data=pd.read_csv("train.csv")
    X,y = Training.Algorithms.scaledata(data)
    detectiontime = data['detection_time']
    models = list()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=False)
    Training.Algorithms.rf(X_train, X_test, y_train, y_test, models)
    Training.Algorithms.dt(X_train, X_test, y_train, y_test, models)
    Training.Algorithms.ada(X_train, X_test, y_train, y_test, models)
    Training.Algorithms.gbt(X, y, X_train, X_test, y_train, y_test, models)
    Training.Algorithms.xgb(X, y, X_train, X_test, y_train, y_test, models)
    Training.Algorithms.lgbm(X, y, X_train, X_test, y_train, y_test, models)
    detectiontime=data['detection_time']
    best, best_name, best_test = Training.Algorithms.compare(models, X_test, y_test)
    return X, y, models, data, best, best_name, best_test, engine

