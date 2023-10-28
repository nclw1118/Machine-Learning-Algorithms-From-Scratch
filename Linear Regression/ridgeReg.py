#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter


class RidgeRegression(object):
    """
     Ridge Regression.
    """
    def __init__(self):
        self.w=None
        self.b=None
    def fit(self, X, y, alpha=0):
        """
        Fits the ridge regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        alpha: regularization parameter.
        """
        # augmented X with extra column of 1's
       	X=np.hstack((X, np.ones((X.shape[0],1))))
        # excluding the bias term!!!
        I=np.identity(X.shape[1])
        I[0,0]=0
        # closed form solution
        self.w=np.linalg.inv(X.T@X+alpha*I)@X.T@y
        
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
        X=np.hstack((X, np.ones((X.shape[0],1))))
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
        return np.sqrt(np.mean((self.predict(X)-y)**2))