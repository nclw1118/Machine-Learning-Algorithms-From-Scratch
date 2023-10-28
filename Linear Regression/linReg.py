#!/usr/bin/python
# -*- coding: utf-8 -*-

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
    def fit(self, X, y):
        """
        Fits the linear regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        """
       	# augmented X with extra column of 1's
       	X=np.hstack((X,np.ones((X.shape[0],1))))
        self.w=np.linalg.inv(X.T@X)@X.T@y
        
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