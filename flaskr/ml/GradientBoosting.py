from sklearn.ensemble import GradientBoostingRegressor
from .Algorithm import Algorithm
import pandas as pd

class GradientBoosting(Algorithm):

    def __init__(self, 
                learning_rate: float, 
                loss: str, 
                max_depth: int, 
                n_estimators: int, 
                subsample: float):
        self.algorithm = GradientBoostingRegressor(learning_rate=learning_rate, 
                                                    loss=loss, 
                                                    max_depth=max_depth, 
                                                    n_estimators=n_estimators, 
                                                    subsample=subsample)

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        self.algorithm.fit(features, targets)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.algorithm.predict(features)
