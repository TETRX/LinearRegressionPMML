from numpy.core.fromnumeric import transpose
import numpy

class RidgeRegressionTrainer:
    def train(self, training_dataset, lambda_):
        X_l,y_l=training_dataset.get_X_y()
        X=numpy.array(X_l,dtype=float)
        y=numpy.array(y_l, dtype=float)
        X_t=X.transpose()
        X_tX=numpy.matmul(X_t,X)
        lambdaI=numpy.eye(training_dataset.y-1)*lambda_
        to_inv=numpy.add(X_tX,lambdaI)
        inv=numpy.linalg.inv(to_inv)
        return numpy.matmul(numpy.matmul(inv,X_t),y).tolist() # (X^TX+lambdaI)^(-1)X^Ty

