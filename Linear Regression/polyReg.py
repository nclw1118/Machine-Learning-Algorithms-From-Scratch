#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd 
from matplotlib import pyplot as plt

from operator import itemgetter


class PolynomialRegression(object):
    """
     Polynomial Regression.
    """
    
    def __init__(self):
        self.w=None
        self.k=None

    def fit(self, X, y, k):
        """
        Fits the polynomial regression model to the training data.

        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        k: polynomial degree
        """
        poly=np.zeros((X.shape[0],k+1))
        for i in range(0,k+1):
            poly[:,i]=np.array(X)**i
            
        self.w=np.linalg.inv(poly.T@poly)@(poly.T)@y
        self.k=k
        

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nx1 matrix of n examples

        Returns
        ----------
        response variable vector for n examples
        """
        poly=np.zeros((X.shape[0],self.k+1))
        for i in range(0,self.k+1):
            poly[:,i]=np.array(X)**i
        return poly@self.w

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
        
        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        
        Returns
        ----------
        RMSE when model is used to predict y
        """
        return np.sqrt(np.mean((self.predict(X)-y)**2))
