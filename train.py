import pandas as pd
import csv
import sklearn
import psycopg2
from sklearn.model_selection import train_test_split
import Algorithms
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
from sqlalchemy import insert, update
from geoalchemy2 import Geography
import sql
import re



#collegamento database
engine = create_engine('postgresql+psycopg2://user:user@localhost:6543/gathering_detection')
connection = engine.connect()
print(engine.table_names())
metadata = MetaData()
gatherings_detection = Table('gatherings_detection', metadata, autoload=True, autoload_with=engine)
gatherings_prediction = Table('gatherings_prediction', metadata, autoload=True, autoload_with=engine)

#import dataset e studio correlazione
data = pd.read_sql_table('gatherings_detection', con=connection)
prediction_df = pd.read_sql_table('gatherings_prediction', con=connection)
Algorithms.heatmap(data)
detectiontime = data['detection_time']

#scaling dati
X,y = Algorithms.scaledata(data)

#list predictions
models = list()

#split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=False)

#random forest
Algorithms.rf(X_train, X_test, y_train, y_test, models)

#decision tree
Algorithms.dt(X_train, X_test, y_train, y_test, models)

#AdaBoost
Algorithms.ada(X_train, X_test, y_train, y_test, models)

#GradientBoostingTree
Algorithms.gbt(X, y, X_train, X_test, y_train, y_test, models)

#Compare score
best, best_name, best_test = Algorithms.compare(models, X, y)

#Predictions to csv
best_test_df=pd.DataFrame()
best_test_df['tracked_point_id']= data['tracked_point_id'].tail(24)
best_test_df['people_concentration'] = best_test
best_test_df['detection_time'] = detectiontime.tail(24)
best_test_df.reset_index(drop=True, inplace=True)

#best_pred = best_test_df.to_csv('best_pred.csv')
res = pd.DataFrame({'Actual': y.tail(24), 'Predicted': best_test})
print(res)
print(best_test)
print(best_test_df.head())

#Send to DB
best_test_df.to_sql('gatherings_prediction', engine, if_exists='append', index=False)

