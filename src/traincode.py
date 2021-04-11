import pandas as pd
from sklearn.model_selection import train_test_split
import Algorithms
import dbConnection

def traincode():
    engine, gatherings_detection, gatherings_prediction, connection = dbConnection.connect()
    #data, prediction_df = dbConnection.dbConnection.get_tables(connection)
    data=pd.read_csv("train.csv")
    x,y = Algorithms.scaledata(data)
    detectiontime = data['detection_time']
    models = list()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=False)
    Algorithms.rf(x_train, x_test, y_train, y_test, models)
    Algorithms.dt(x_train, x_test, y_train, y_test, models)
    Algorithms.gbt(x_train, x_test, y_train, y_test, models)
    Algorithms.xgb(x_train, x_test, y_train, y_test, models)
    detectiontime=data['detection_time']
    best, best_name, best_test = Algorithms.compare(models, x_test, y_test)
    return x, y, models, data, best, best_name, best_test, engine

