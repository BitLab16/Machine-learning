from .GradientBoosting import GradientBoosting
from .AlgorithmFactory import AlgorithmFactory

class GradientBoostingFactory(AlgorithmFactory):

    def create(self) -> GradientBoosting:
        return GradientBoosting(learning_rate= 0.01, 
                                loss= 'ls', 
                                max_depth= 8, 
                                n_estimators= 150, 
                                subsample= 0.2)