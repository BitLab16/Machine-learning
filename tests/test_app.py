import pytest
from datetime import datetime

from app import to_date

def test_to_date():
    date = '2018-12-14 03:00'
    datetime_data = to_date(date)
    assert isinstance(datetime_data, datetime)
    assert datetime_data == datetime(2018,12,14,3,0)
