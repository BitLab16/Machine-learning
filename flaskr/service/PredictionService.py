from injector import inject

from datetime import datetime

from ..ml import Algorithm
from ..ml import Scaler
from ..repository import Db
from ..model import TrackedPoint
from ..model import Prediction

'''
Classe che si occupera di effettuare le predizioni esponendole come un servizio
'''
class PredictionService:

    @inject
    def __init__(self, model: Algorithm, db: Db, scaler: Scaler):
        self.model = model
        self.db = db
        self.scaler = scaler


    '''
    time_from => timestamp da cui cominciare le predizioni
    time_to => timestamp in cui finire le predizioni
    '''
    def predict(self, time_from: datetime, time_to: datetime):
        id_list = self.db.get_different_point_id()
        result = []
        for id in id_list:
            result.append(self.predict_for_point(id, time_from, time_to))
        return result


    def predict_for_point(self, id: int, time_from: datetime, time_to: datetime) -> TrackedPoint:
        features= self.db.get_feature_from_interval(id, time_from, time_to)
        features, times = self.scaler.scale_for_prediction(features)
        predictions = self.model.predict(features)
        tracked_point = TrackedPoint(id)
        for i in range(0, predictions.size):
            a = times.iloc[i]
            b = predictions[i]
            prediction = Prediction(a, b)
            tracked_point.add_predictions(prediction)
        return tracked_point

