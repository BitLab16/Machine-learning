import pytest
import pandas as pd
from datetime import datetime

from flaskr.ml import Algorithm, DecisionTree, GradientBoosting, RandomForest

'''
def features_for_testing(data_for_testing):
    test_features = data_test.drop(["people_concentration"], axis=1)
    return test_features

def targets_for_testing(data_for_testing):
    test_targets = data_test["people_concentration"]
    return test_targets


@pytest.fixture(scope='module')
def new_dtmodel(features_for_testing, targets_for_testing):
    dt_model = DecisionTree()
    dt_model.fit(test_features, test_targets)
    return dt_model
    
def new_rfmodel(features_for_testing, targets_for_testing):
    rf_model = RandomForest()
    rf_model.fit(test_features, test_targets)
    return rf_model
    
def gb_newmodel(features_for_testing, targets_for_testing):
    gb_model = GradientBoosting()
    gb_model.fit(test_features, test_targets)
    return gb_model
    '''
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
    data_for_testing = pd.read_csv('test_dataset.csv')
    return data_for_testing
