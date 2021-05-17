import pytest
import pandas as pd
import json
from unittest import TestCase

from flaskr.model.Prediction import Prediction
from flaskr.ml.DecisionTree import DecisionTree
'''
def test_to_json(data_for_testing):
    data = data_for_testing.loc[(data_for_testing['detection_time'] == '2018-01-10 14:05') & (data_for_testing['tracked_point_id'] == 1)]
    expected_result = {'time': ['2018-01-10 14:05'], 'flow': [8]}
    prediction = Prediction(data['detection_time'], data['people_concentration'])
    result = prediction.to_json()
    assert result == expected_result
'''