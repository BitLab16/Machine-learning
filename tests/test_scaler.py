import pytest
import pandas as pd
import numpy as np
from flaskr.ml.Scaler import Scaler
from datetime import datetime

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
'''
def test__common_scale(data_for_testing):
    scaler = Scaler()
    data = scaler._common_scale(data_for_testing)
    assert 'time' in data.columns
    assert 'date' in data.columns
    assert 'holiday' in data.columns
'''


