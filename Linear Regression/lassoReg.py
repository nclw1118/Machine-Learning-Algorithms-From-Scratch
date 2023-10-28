#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter


class LassoRegression(object):
    """
     Lasso Regression with gradient descent.
    """
    def __init__(self):
        self.w=None
        self.b=None
    def fit(self, X, y, alpha=0, eta=0.01, max_iter=1000, min_diff=1e-05):
        """
        Fits the lasso regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: nx1 response variable vector for n examples
        alpha: regularization parameter
        eta: step-size for gradient descent
        max_iter: maximum number of iterations for gradient descent
        min_diff: minimum difference of loss for early stop
        """
        # augmented X with extra column of 1's
        N,D=X.shape
        
        #gradient descent
        w_=np.random.rand(D,1)
        b_=np.random.rand()
        prev_cost=np.inf
        iterations=0
        while iterations<max_iter:
            dw=X.T@(X@w_+b_-y)+np.sign(w_)*alpha
            db=np.sum(X@w_+b_-y)
            w_=w_-eta*dw
            b_=b_-eta*db
            cur_cost=np.linalg.norm(X@w_+b_-y)**2
            if np.abs(cur_cost-prev_cost)<min_diff:
                break
            prev_cost=cur_cost
            iterations+=1
        if iterations <max_iter-1:
            print("early stop at iter:"+str(iterations),", cost=", str(cur_cost))
        self.w=w_
        self.b=b_
        
        
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
        return X@self.w+self.b

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
        return np.sqrt(np.mean((self.predict(X)-y)**2))