import abc
from math import cos
from random import randint
from ..derivative_searcher import DerivativeSearcher
from ..a_regression_trainer import RegressionTrainer

class CoordinateRegressionTrainer(RegressionTrainer):
    def train(self, training_dataset, lambdas):
        X,y=training_dataset.get_X_y()
        theta=[0 for j in range(len(X[0]))]
        for i in range(10*len(X[0])):
            j=randint(0,len(X[0])-1)
            derivative_searcher=DerivativeSearcher(self.get_cost_func(theta,X,y,j,lambdas))
            theta[j]=derivative_searcher.minimalize(training_dataset,theta,j,lambdas)
        return theta


    @abc.abstractmethod
    def get_cost_func(self, theta,X, y,j,lambdas):
        pass
