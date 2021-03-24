import pandas as pd
from sklearn.model_selection import train_test_split
import Training.Algorithms
import dbConnection.dbConnection

def predictions(X, y, models, data, engine):
    detectiontime=data['detection_time']
    best, best_name, best_test = Training.Algorithms.compare(models, X, y)
    best_test_df=pd.DataFrame()
    best_test_df['tracked_point_id']= data['tracked_point_id'].tail(24)
    best_test_df['people_concentration'] = best_test
    best_test_df['detection_time'] = detectiontime.tail(24)
    best_test_df.reset_index(drop=True, inplace=True)

    res = pd.DataFrame({'Actual': y.tail(24), 'Predicted': best_test})
    print(res)
    print(best_test)
    print(best_test_df.head())

    best_test_df.to_sql('gatherings_prediction', engine, if_exists='append', index=False)