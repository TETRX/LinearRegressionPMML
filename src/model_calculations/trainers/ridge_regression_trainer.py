from numpy.core.fromnumeric import transpose
import numpy

class RidgeRegressionTrainer:
    def __init__(self):
        self.LAMBDAS_NUM=1

    def train(self, training_dataset, lambdas):
        lambda_=lambdas[0]
        X_l,y_l=training_dataset.get_X_y()
        X=numpy.array(X_l,dtype=float)
        y=numpy.array(y_l, dtype=float)
        X_t=X.transpose()
        X_tX=numpy.matmul(X_t,X)
        lambdaI=numpy.eye(training_dataset.y-1)*lambda_
        to_inv=numpy.add(X_tX,lambdaI)
        inv=numpy.linalg.inv(to_inv)
        return numpy.matmul(numpy.matmul(inv,X_t),y).tolist() # (X^TX+lambdaI)^(-1)X^Ty

    def cost(self, dataset,theta, lambdas):
        lambda_=lambdas[0]
        sum_all=0
        X,y=dataset.get_X_y()
        m=len(X)
        for k in range(m):
            sum_k=0
            for i in range(len(X[0])):
                sum_k+=theta[i]*X[k][i]
            sum_all+=(sum_k-y[k])**2
        mean_all=1/m*sum_all

        norm=0
        for i in range(len(theta)):
            norm+=(theta[i])**2
        return mean_all+lambda_*norm