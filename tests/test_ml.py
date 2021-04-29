import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from datetime import datetime

from flaskr.ml.Scaler import Scaler
from flaskr.ml.Trainer import Trainer
from flaskr.ml import Algorithm, DecisionTreeFactory, GradientBoostingFactory, RandomForestFactory, DecisionTree, GradientBoosting, RandomForest
from flaskr.repository.FileReader import FileReader

def test__split_date(datetime_test):
    scaler = Scaler()
    date, time = scaler._split_date(datetime_test)
    assert date[0] == 20171128
    assert time[0] == 2355

def test__remove_unused_data(datetimepd_test, timearray_test):
    scaler = Scaler()
    data = scaler._remove_unused_data(timearray_test, datetimepd_test)
    timearray_test = pd.Series(timearray_test)
    assert data['time'].all() in timearray_test

def test__common_scale(data_for_testing):
    scaler = Scaler()
    data = scaler._common_scale(data_for_testing)
    assert 'time' in data.columns
    assert 'date' in data.columns
    assert 'holiday' in data.columns
    assert 'detection_time' not in data.columns

def test_scale(data_for_testing):
    scaler = Scaler()
    x, y = scaler.scale(data_for_testing)
    y = pd.DataFrame(y)
    print(type(x))
    print(type(y))
    assert set(['season','weather', 'events', 'attractions', 'holiday', 'weather_index', 'attractions_index', 'event_index', 'time', 'date']).issubset(x.columns)
    assert 'people_concentration' not in x.columns
    assert 'people_concentration' in y

def test_scale_for_prediction(data_for_testing):
    scaler = Scaler()
    x, detection_time = scaler.scale_for_prediction(data_for_testing)
    print(type(detection_time))
    assert set(['season','weather', 'events', 'attractions', 'holiday', 'weather_index', 'attractions_index', 'event_index', 'time', 'date']).issubset(x.columns)
    assert 'people_concentration' not in x.columns
    assert 'detection_time' not in x.columns
    assert data_for_testing['detection_time'].all() == detection_time.all()

def test_train(data_for_testing, prediction_data_for_testing):
    decision_tree = DecisionTreeFactory().create()
    gradient_boosting = GradientBoostingFactory().create()
    random_forest = RandomForestFactory().create()
    models = [decision_tree, gradient_boosting, random_forest]
    scaler = Scaler()
    features_for_testing, targets_for_testing = scaler.scale(data_for_testing)
    trainer = Trainer(0.2, features_for_testing, targets_for_testing)
    trainer.train(models)
    x, detection_time = scaler.scale_for_prediction(prediction_data_for_testing)
    for model in models:
        try:
            model.predict(x)
        except NotFittedError as e:
            print(repr(e))


