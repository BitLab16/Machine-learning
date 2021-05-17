from .DecisionTree import DecisionTree
from .AlgorithmFactory import AlgorithmFactory

class DecisionTreeFactory(AlgorithmFactory):

    def create(self) -> DecisionTree:
        return DecisionTree(max_depth = 20, 
                            min_samples_leaf = 10, 
                            min_samples_split = 500, 
                            min_weight_fraction_leaf = 0.0, 
                            random_state = None, 
                            splitter = 'random', 
                            max_features = 'auto')