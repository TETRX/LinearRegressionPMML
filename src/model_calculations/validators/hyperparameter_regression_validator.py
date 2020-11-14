from abc import abstractmethod
from ..a_regression_validator import RegressionValidator
from itertools import product
import math

class HyperparameterRegressionValidator(RegressionValidator):
    def __init__(self, training_dataset, validation_dataset,possible_params,lambda_val=[0,0.1,0.2,0.4,0.8,1.6]):
        super().__init__(training_dataset, validation_dataset,lambda_val=lambda_val)
        self.POSSIBLE_PARAMS=possible_params

    def get_best_hyper(self, trainer):
        all_lambda_comb=[a for a in product(self.LAMBDA_VALUES,repeat=trainer.LAMBDAS_NUM)]
        all_param_comb=[a for a in product(self.POSSIBLE_PARAMS,repeat=self.training_dataset.y)]
        best_hyper=(all_lambda_comb[0],all_param_comb[0])
        best_error=math.inf
        for lambda_comb in all_lambda_comb:
            for param_comb in all_param_comb:
                basis_functions=self.get_basis_func(param_comb)
                print("polynomial degrees: ")
                print(param_comb)
                print("lambdas: ")
                print(lambda_comb)
                functioned_dataset=self.training_dataset.apply_functions(basis_functions)
                print("appled functions")
                curr_theta=trainer.train(functioned_dataset,lambda_comb)
                print("theta:")
                print(curr_theta)
                functioned_val_dataset=self.validation_dataset.apply_functions(basis_functions)
                error=self.validate(trainer,curr_theta,functioned_val_dataset,lambda_comb)
                print("error:")
                print(error)
                if error<best_error:
                    best_error=error
                    best_hyper=(lambda_comb,param_comb)
                break
            break
        return (best_hyper[0],self.get_basis_func(best_hyper[1]))
        
    def validate(self,trainer,theta,validation_dataset,lambdas_):
        return trainer.cost(validation_dataset,theta,lambdas_)

    @abstractmethod
    def get_basis_func(self,degrees):
        pass