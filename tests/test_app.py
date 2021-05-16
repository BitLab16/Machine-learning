import pytest
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

from app import to_date, compare_algorithm, train_ml
from flaskr.ml import Algorithm, DecisionTree, GradientBoosting, RandomForest
from flaskr.ml import DecisionTreeFactory
from flaskr.ml import GradientBoostingFactory
from flaskr.ml import RandomForestFactory
from flaskr.ml import Scaler
from flaskr.ml import Trainer
from flaskr.service import PredictionService
from flaskr.repository import FileReader
from flaskr.repository import Db


def test_to_date():
    date = '2018-12-14 03:00'
    datetime_data = to_date(date)
    assert isinstance(datetime_data, datetime)
    assert datetime_data == datetime(2018,12,14,3,0)

def test_compare_algorithm(decision_tree_factory, random_forest_factory, gradient_boosting_factory, features_test_splitted, targets_test_splitted):
    decision_tree = decision_tree_factory.create() 
    random_forest = random_forest_factory.create() 
    gradient_boosting = gradient_boosting_factory.create() 
    x_train, x_test, y_train, y_test = train_test_split(features_test_splitted, targets_test_splitted, test_size=0.2, shuffle=False, random_state=False)
    decision_tree.fit(x_train, y_train)
    random_forest.fit(x_train, y_train)
    gradient_boosting.fit(x_train, y_train)
    models = []
    models.append(decision_tree)
    models.append(random_forest)
    models.append(gradient_boosting)
    best, best_test = compare_algorithm(models, features_test_splitted, targets_test_splitted)
    assert isinstance(best_test, Algorithm)
    assert best > 0

def test_train_ml(features_test_splitted, targets_test_splitted):
    x_train, x_test, y_train, y_test = train_test_split(features_test_splitted, targets_test_splitted, test_size=0.2, shuffle=False, random_state=False)
    x_test = x_test.drop('date', axis=1)
    model = train_ml()
    try:
        model.predict(x_test)
    except NotFittedError as e:
        print(repr(e))
    assert isinstance(model, Algorithm)
    assert pd.notnull(model)

def test_prediction_for_single_point_in_interval(app, client):
    route = '/prediction/1/?from=2021-03-07 08:00&to=2021-03-07 12:00'
    rv = client.get(route)
    assert rv.status_code == 200

def test_prediction_for_all_point_in_interval(app, client):
    route = '/prediction/?from=2021-03-07 08:00&to=2021-03-07 12:00'
    rv = client.get(route)
    assert rv.status_code == 200
