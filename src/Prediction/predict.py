import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Training.Algorithms import scaledata

def predictions(X, y, models, data, best, best_name, best_test, engine):
    best_test_df=pd.read_csv("predict_demo.csv")
    detectiontime = best_test_df["detection_time"]
    trackedpointid = best_test_df["tracked_point_id"]
    X, y = scaledata(best_test_df)
    pred = best_test.predict(X)
    best_test_df = best_test_df.drop(["season", "holiday", "weather", "events", "attractions", "weather_index", "attractions_index", "event_index", "time_index", "date", "time"], axis=1)
    best_test_df["detection_time"] = detectiontime
    best_test_df["tracked_point_id"] = trackedpointid
    res = pd.DataFrame({'Actual': y, 'Predicted '+best_name: pred})
    print(res)
    print(best_test)
    print(best_test_df.head())
    best_test_df["id"] = np.arange(1, len(best_test_df) + 1)
    #best_test_df.reset_index(drop=True, inplace=True)
    best_test_df.to_sql('gatherings_prediction', engine, if_exists='append', index=False)

