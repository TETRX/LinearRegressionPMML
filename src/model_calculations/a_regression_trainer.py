import abc
from math import cos
from random import randint
class RegressionTrainer:

    @abc.abstractmethod
    def train(self, training_dataset, lambdas):
        pass


    @abc.abstractmethod
    def get_cost_func(self, theta,X, y,j):
        pass
    def cost(self, dataset,theta):
        sum_all=0
        X,y=dataset.get_X_y()
        m=len(X)
        for k in range(m):
            sum_k=0
            for i in range(len(X[0])):
                sum_k+=theta[i]*X[k][i]
            sum_all+=(sum_k-y[k])**2
        mean_all=1/m*sum_all

        return mean_all