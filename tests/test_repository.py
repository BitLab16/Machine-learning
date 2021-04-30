import pytest
import pandas as pd
import numpy as np

from datetime import datetime

from flaskr.repository.Db import Db
from flaskr.repository.FileReader import FileReader

def test_get_feature_from_interval(data_for_testing, db_test):
    result = db_test.get_feature_from_interval(1, '2018-01-01 00:00', '2018-01-21 19:55')
    id_test = [1]
    assert result['people_concentration'].size == data_for_testing['people_concentration'].loc[data_for_testing['tracked_point_id'].isin(id_test)].size


def test_get_different_point_id(data_for_testing, db_test):
    id_list = db_test.get_different_point_id()
    for i in range(0, 5, 1):
        print(i)
        assert id_list[i] == (i+1)


