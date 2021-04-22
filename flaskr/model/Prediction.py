
class Prediction:

    def __init__(self, time, flow):
        self.time = time
        self.flow = flow

    def to_json(self) ->dict:
        return {
            "time": self.time,
            "flow": self.flow
        } 