from sklearn.ensemble import RandomForestRegressor
from .Algorithm import Algorithm
import pandas as pd

class RandomForest(Algorithm):

    def __init__(self, 
                bootstrap: bool, 
                max_depth: int, 
                max_features: str, 
                min_samples_leaf: int, 
                min_samples_split: int, 
                n_estimators: int, 
                random_state: int):
        self.algorithm = RandomForestRegressor(bootstrap=bootstrap, 
                                                max_depth=max_depth, 
                                                max_features=max_features, 
                                                min_samples_leaf=min_samples_leaf, 
                                                min_samples_split=min_samples_split, 
                                                n_estimators=n_estimators, 
                                                random_state=random_state)

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        self.algorithm.fit(features, targets)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.algorithm.predict(features)
