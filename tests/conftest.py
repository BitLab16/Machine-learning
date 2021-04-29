import pytest
import pandas as pd
from injector import inject
from pathlib import Path

from datetime import datetime

from flaskr.ml import Algorithm, DecisionTree, GradientBoosting, RandomForest
from flaskr.repository import FileReader

@pytest.fixture(scope='function')
def datetime_test():
    d = {'a': "2017-11-28 23:55", 'b': '2017-11-28 23:30'}
    datetime_test = pd.Series(data=d, index=['a', 'b'])
    return datetime_test

@pytest.fixture(scope='function')
def datetimepd_test():
    datetimepd = {'date': [20171128, 20171128, 20171128, 20171128], 'time':[2300, 2305, 2330, 2345]}
    datetimepd_test = pd.DataFrame(datetimepd)
    return datetimepd_test

@pytest.fixture(scope='function')
def timearray_test():
    timearray_test = [0, 30, 100, 130, 200, 230, 300, 330, 400, 430, 500, 530,
             600, 630, 700, 730, 800, 830, 900, 930, 1000, 1030, 1100,
             1130, 1200, 1230, 1300, 1330, 1400, 1430, 1500, 1530,
             1600, 1630, 1700, 1730, 1800, 1830, 1900, 1930, 2000,
             2030, 2100, 2130, 2200, 2230, 2300, 2330]
    return timearray_test

@pytest.fixture(scope='function')
def data_for_testing():
    file_reader = FileReader('test_dataset.csv')
    data_for_testing = file_reader.read_file()
    return data_for_testing

@pytest.fixture(scope='function')
def prediction_data_for_testing():
    file_reader = FileReader('prediction_test_dataset.csv')
    prediction_data_for_testing = file_reader.read_file()
    return prediction_data_for_testing


