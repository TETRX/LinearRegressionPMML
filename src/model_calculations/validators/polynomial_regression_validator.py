from .hyperparameter_regression_validator import HyperparameterRegressionValidator
from ..a_regression_trainer import RegressionTrainer

class PolynomialRegressionValidator(HyperparameterRegressionValidator):

    def __init__(self, training_dataset, validation_dataset,degrees=[1,2,3,4,5],lambda_val=[0,0.1,0.4,1.6]):
        super().__init__(training_dataset,validation_dataset,degrees,lambda_val=lambda_val)

    def get_basis_func(self, params):
        def get_power(degree):
            return lambda x: x**degree
        return [get_power(degree) for degree in params]