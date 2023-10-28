import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter


class LinearRegression(object):
    """
     Linear Regression with closed form solution.
    """
    def __init__(self):
        self.w=None
    def fit(self, X, y, alpha, max_iter=1000, min_diff=1e-05):
        """
        Fits the linear regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        alpha: step size for gradient descent
        max_iter: max number of iterations in gradient descent
        min_diff: minimum update size for early stop
        """
        N,D=X.shape[0],X.shape[1]
        X=np.hstack([X,np.ones((N,1))])
        
        #gradient descent
        #randomly initialize w
        w_=np.random.rand(D+1,1)
        iteration=0
        prev_cost=np.inf
        #stop when exceed max iteration and can't reach minimum update requirement
        while iteration<max_iter:
            gradient=X.T@(X@w_ - y)/N
            w_-=alpha*gradient
            cur_cost=np.linalg.norm(X@w_-y)**2/N
            if np.abs(cur_cost-prev_cost)<min_diff:
                break
            prev_cost=cur_cost
            iteration+=1
        if iteration <max_iter-1:
            print("early stop at iter:"+str(iteration),", cost=", str(cur_cost))
        self.w=w_
        
    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates

        Returns
        ----------
        response variable vector for n examples
        """
        # augment test input X
        X=np.hstack((X,np.ones((X.shape[0],1))))
        return X@self.w

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
            
        Arguments
        ----------
        X: nxp matrix of n examples with p covariates
        y: response variable vector for n examples
            
        Returns
        ----------
        RMSE when model is used to predict y
        """
       	# Your code here
        return np.sqrt(np.mean((self.predict(X)-y)**2))