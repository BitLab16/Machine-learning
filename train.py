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
#Algorithms.heatmap(data)
"""data['new_date'] = [d.date() for d in data['detection_time']]
data['new_time'] = [d.time() for d in data['detection_time']]
data.replace(regex=['-'], value='', inplace=True)"""
data['holiday'] = data.holiday.astype(int)
detectiontime = data['detection_time']
data=data.drop(["detection_time"], axis=1)
with pd.option_context('display.max_columns', None, 'display.max_rows', None):
 print(data.head())

#scaling dati
X,y = Algorithms.scaledata(data)

#list predictions
models = list()

#split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
"""
#Send to DB
predictions = (
    gatherings_prediction.insert(None).values(id='003', tracked_point_id='001', detection_time='2021-3-5 17:00:00', people_concentration='10') #people_concentration = best_test
)
engine.execute(predictions)
"""
best_test_df.to_sql('gatherings_prediction', engine, if_exists='append', index=False)

