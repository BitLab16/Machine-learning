from .RandomForest import RandomForest
from .AlgorithmFactory import AlgorithmFactory

class RandomForestFactory(AlgorithmFactory):

    def create(self) -> RandomForest:
        return RandomForest(bootstrap = True, 
                            max_depth = 10, 
                            max_features = 'log2', 
                            min_samples_leaf = 5, 
                            min_samples_split = 5, 
                            n_estimators = 100, 
                            random_state = None)