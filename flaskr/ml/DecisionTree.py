from sklearn.tree import DecisionTreeRegressor
from .Algorithm import Algorithm
import pandas as pd


class DecisionTree(Algorithm):

    def __init__(self, 
                max_depth: int, 
                min_samples_leaf: int, 
                min_samples_split: int, 
                min_weight_fraction_leaf: int, 
                random_state: int, 
                splitter: str, 
                max_features: str):
                
        
        self.algorithm = DecisionTreeRegressor(max_depth=max_depth, 
                                                min_samples_leaf=min_samples_leaf, 
                                                min_samples_split=min_samples_split, 
                                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                random_state =random_state, 
                                                splitter=splitter, 
                                                max_features=max_features)

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        self.algorithm.fit(features, targets)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.algorithm.predict(features)
