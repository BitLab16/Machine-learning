import pandas as pd

class Scaler:

    def __init__(self):
        self.time_array = [0, 30, 100, 130, 200, 230, 300, 330, 400, 430, 500, 530,
             600, 630, 700, 730, 800, 830, 900, 930, 1000, 1030, 1100,
             1130, 1200, 1230, 1300, 1330, 1400, 1430, 1500, 1530,
             1600, 1630, 1700, 1730, 1800, 1830, 1900, 1930, 2000,
             2030, 2100, 2130, 2200, 2230, 2300, 2330]


    def _split_date(self, data: pd.Series) -> pd.Series:
        date_time_splitted = data.str.split(" ", n = 0, expand = True)
        date = date_time_splitted[0]
        time = date_time_splitted[1]
        date = pd.Series(date.astype(str).str.replace('-', '', regex=False))
        time = pd.Series(time.astype(str).str.replace(':', '', regex=False))
        date = date.astype(int)
        time = time.astype(int)
        return date, time


    def _remove_unused_data(self, time_array: list, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[data['time'].isin(time_array)]
        return data


    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._common_scale(data)
        data = self._remove_unused_data(self.time_array, data)
        x = data.drop(["people_concentration"], axis=1)  # features
        y = data["people_concentration"]
        return x, y

    def scale_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        date, time = self._split_date(data["detection_time"].astype('str'))
        detection_time = data.detection_time
        data = data.drop(["detection_time","tracked_point_id"], axis=1)
        data['holiday'] = data.holiday.astype(int)
        data["date"] = date
        data["time"] = time
        x = data.drop(["people_concentration"], axis=1)
        return x, detection_time

    def _common_scale(self, data: pd.DataFrame) -> pd.DataFrame:
        a = data['detection_time']
        date, time = self._split_date(a)
        data = data.drop(["detection_time","tracked_point_id"], axis=1)
        data['holiday'] = data.holiday.astype(int)
        data["date"] = date
        data["time"] = time
        return data