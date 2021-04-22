import pandas as pd
import abc

class Algorithm(metaclass=abc.ABCMeta):
    """
    L'interfaccia Algorithm dichiara le operazioni che ogni prodotto concreto
    deve implementare
    """
    
        
    @abc.abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        pass
    

    @abc.abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        pass