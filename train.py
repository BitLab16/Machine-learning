import pandas as pd
import csv
import sklearn
import psycopg2
from sklearn.model_selection import train_test_split
import Algorithms
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table
from sqlalchemy import insert, update

#collegamento database
engine = create_engine('postgresql+psycopg2://user:user@localhost:6543/gathering_detection')
connection = engine.connect()
print(engine.table_names())
metadata = MetaData()
gatherings_detection = Table('gatherings_detection', metadata, autoload=True, autoload_with=engine)
gatherings_prediction = Table('gatherings_prediction', metadata, autoload=True, autoload_with=engine)

#import dataset e studio correlazione
detection_df = pd.read_sql_table('gatherings_detection', con=connection)
prediction_df = pd.read_sql_table('gatherings_prediction', con=connection)
df = pd.read_csv("https://raw.githubusercontent.com/Cionsa/Datasets/main/hour.csv", delimiter=',')
data = df.drop(['instant', 'registered', 'casual', 'dteday'], axis=1)
Algorithms.heatmap(data)

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
best, best_name, best_test = Algorithms.compare(models, X_test, y_test)

#Predictions to csv
best_test_df = pd.DataFrame(data = best_test)
best_pred = best_test_df.to_csv('best_pred.csv')
res = pd.DataFrame({'Actual': y_test, 'Predicted': best_test})
print(res)

#Send to DB
predictions = (
    gatherings_prediction.insert(None).values(id='003', tracked_point_id='001', detection_time='2021-3-5 17:00:00', people_concentration='10') #people_concentration = best_test
)
engine.execute(predictions)


#Invio csv al db

"""
cur = connection.cursor()
with open('best_pred.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader) # Skip the header row.
    for row in reader:
        cur.execute(
        "INSERT INTO users VALUES (%s, %s, %s, %s)", row
    )
connection.commit()

gatherings_prediction.insert(None).values(id={gatherings_prediction.pg_total_relation_size()+i}, tracked_point_id={X_test[i][1]}, detection_time={X_test[i][2]}, people_concentration={y_test[i]})

"""