import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        # class labels
        self.classes=None
        # priors P(y=c) for each class c
        self.prior={}
        # mean of each feature xj for each class c
        self.mean={}
        # variance of each feature xj for each lass c
        self.var={}

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier.
        
        Parameters:
        - X: Training data, shape (n_samples, n_features)
        - y: Target values, shape (n_samples,)
        """
        self.classes=np.unique(y)
        for c in self.classes:
            self.prior[c]=np.mean(y==c)
            self.mean[c]=np.mean(X[y==c],axis=0)
            self.var[c]=np.var(X[y==c],axis=0)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        - X: Samples, shape (n_samples, n_features)
        
        Returns:
        - Predicted class labels, shape (n_samples,)
        """
        preds=[]
        for x in X:
            best_prob=-np.inf
            pred=None
            for c in self.classes:
                # log likelihood = log prior + log sum of guassian of each feature
                prob=np.log(self.prior[c]) + np.sum(np.log(self.gaussian_pdf(x,self.mean[c],self.var[c])))
                if prob>best_prob:
                    best_prob=prob
                    pred=c
            preds.append(pred)
        return np.array(preds)

    def gaussian_pdf(self, x, mean, var):
        """
        Compute the Gaussian probability density function.
        
        Parameters:
        - x: Value
        - mean: Mean of the Gaussian
        - var: Variance of the Gaussian
        
        Returns:
        - Probability density value
        """
        return (1.0/np.sqrt(2*np.pi*var))*np.exp(-(x-mean)** 2/(2*var))

