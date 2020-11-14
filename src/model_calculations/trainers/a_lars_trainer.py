import abc
from random import randint
from ..ternary_searcher import TernarySearcher

class LarsRegressionTrainer:
    ERROR=0.001
    def train(self, training_dataset, lambdas):
        X,y=training_dataset.get_X_y()
        theta=[0 for j in range(len(X[0]))]
        for i in range(10*len(X[0])):
            j=randint(0,len(X[0])-1)
            ternary_searcher=TernarySearcher(self.get_cost_func(theta,X,y,j,lambdas),LarsRegressionTrainer.ERROR)
            theta[j]=ternary_searcher.minimalize()
        return theta


    @abc.abstractmethod
    def get_cost_func(self, theta,X, y,j,lambdas):
        pass

    @abc.abstractmethod
    def cost(self, dataset,theta, lambdas):
        pass