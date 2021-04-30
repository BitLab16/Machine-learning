import pandas as pd
from injector import inject

from datetime import datetime

from .FileReader import FileReader

class Db:

    @inject
    def __init__(self, file_reader: FileReader):
        self.file_reader = file_reader
        self.data_frame = self.file_reader.read_file()
        self.data_frame["detection_time"] = pd.to_datetime( self.data_frame["detection_time"])  

    def get_feature_from_interval(self, id: int, time_from: datetime, time_to: datetime) -> pd.DataFrame:
        mask = (self.data_frame["detection_time"] >= time_from) & (self.data_frame["detection_time"] <= time_to) & (self.data_frame.tracked_point_id == id)
        result = self.data_frame.loc[mask]
        return result

    def get_different_point_id(self): 
        return self.data_frame.tracked_point_id.unique()
