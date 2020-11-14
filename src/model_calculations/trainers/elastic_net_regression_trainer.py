from LinearRegressionPMML.src.model_calculations.trainers.a_lars_trainer import LarsRegressionTrainer
from numpy.core.fromnumeric import transpose
import numpy

class ElasticNetRegressionTrainer(LarsRegressionTrainer):
    def __init__(self):
        self.LAMBDAS_NUM=2
        
    def get_cost_func(self, theta,X, y,j,lambdas):
        lambda_1=lambdas[0]
        lambda_2=lambdas[1]
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
                sum_all+=(sum_k-y[k][0])**2
            mean_all=1/m*sum_all

            norm_1=0
            norm_2=0
            for i in range(len(theta)):
                if i==j:
                    norm_2+=theta_j**2
                    norm_1+=abs(theta_j)
                else:
                    norm_2+=theta[i]**2
                    norm_1+=abs(theta[i])
            return mean_all+lambda_1*norm_1+lambda_2*norm_2
        return cost_func

    def cost(self, dataset,theta, lambdas):
        lambda_1=lambdas[0]
        lambda_2=lambdas[1]
        sum_all=0
        X,y=dataset.get_X_y()
        m=len(X)
        for k in range(m):
            sum_k=0
            for i in range(len(X[0])):
                sum_k+=theta[i]*X[k][i]
            sum_all+=(sum_k-y[k])**2
        mean_all=1/m*sum_all

        norm_1=0
        norm_2=0
        for i in range(len(theta)):
            norm_2+=theta[i]**2
            norm_1+=abs(theta[i])
        return mean_all+lambda_1*norm_1+lambda_2*norm_2