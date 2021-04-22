import abc

class AlgorithmFactory(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self):
        pass