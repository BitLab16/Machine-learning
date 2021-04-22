from .Prediction import Prediction

class TrackedPoint:

    def __init__(self, point_id: int):
        self.point_id = point_id
        self.predictions = []

    def add_predictions(self, predictions_list) -> None: 
        self.predictions.append(predictions_list)

    def to_json(self) -> dict:
        predictions = []
        for item in self.predictions:
            predictions.append(item.to_json)
        return {
            "point": self.point_id,
            "prediction": predictions
        }

    