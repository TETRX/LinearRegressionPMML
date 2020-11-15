import math

class DerivativeSearcher:

    def __init__(self, to_min):
        self.to_min=to_min

    def minimalize(self,dataset,theta,i, lambdas):
        lambda_1=0 if len(lambdas)==1 else lambdas[1]
        lambda_2=lambdas[-1]
        X,y=dataset.get_X_y()
        a_i=0
        for x in X:
            a_i+=2*x[i]**2
        a_i/=len(X)
        a_i+=2*lambda_1

        c_i=0
        for l in range(len(X)):
            x=X[l]
            summ=0
            for j in range(len(x)):
                if j==i:
                    continue
                summ-=theta[j]*x[j]
            summ+=y[l]
            summ*=2*x[i]
            c_i+=summ
        c_i/=len(X)

        # if theta_i>0
        theta_i_pos=(c_i+lambda_2)/a_i

        if theta_i_pos<0:
            theta_i_pos=0

        # if theta_i<0
        theta_i_neg=(c_i-lambda_2)/a_i

        if theta_i_neg>0:
            theta_i_neg=0

        return theta_i_pos if self.to_min(theta_i_pos)<self.to_min(theta_i_neg) else theta_i_neg