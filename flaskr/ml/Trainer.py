import pandas as pd
from sklearn.model_selection import train_test_split

class Trainer:

    def __init__(self, test_size: float, features: pd.DataFrame, targets: pd.DataFrame):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, targets, test_size=test_size, shuffle=False, random_state=False)

    def train(self, models: list) -> None:
        for model in models:
            model.fit(self.x_train, self.y_train)