import abc
from abc import abstractmethod
import random
from random import randint

class RegressionValidator:
    def __init__(self, training_dataset, validation_dataset,lambda_val=[0,0.1,0.2,0.4,0.8,1.6]):
        self.training_dataset=training_dataset
        self.validation_dataset=validation_dataset
        self.LAMBDA_VALUES=lambda_val

    @abstractmethod
    def get_best_hyper(self,trainer): #return (regression_trainer,basis_functions)
        pass