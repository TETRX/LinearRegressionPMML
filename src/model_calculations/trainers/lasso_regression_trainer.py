from .coordinate_regression_trainer import CoordinateRegressionTrainer
from numpy.core.fromnumeric import transpose
import numpy

class LassoRegressionTrainer(CoordinateRegressionTrainer):
    def __init__(self):
        self.LAMBDAS_NUM=1
    def get_cost_func(self, theta,X, y,j,lambdas):
        lambda_=lambdas[0]
        def cost_func(theta_j):
            sum_all=0
            m=len(X)
            for k in range(m):
                sum_k=0
                for i in range(len(X[0])):
                    if i==j:
                        sum_k+=theta_j*X[k][i] 
                    else:
                        sum_k+=theta[i]*X[k][i]
                sum_all+=(sum_k-y[k])**2
            mean_all=1/m*sum_all

            norm=0
            for i in range(len(theta)):
                if i==j:
                    norm+=abs(theta_j)
                else:
                    norm+=abs(theta[i])
            return mean_all+lambda_*norm
        return cost_func
