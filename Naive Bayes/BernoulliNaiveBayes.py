import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, alpha=1):
        # smoothing paramter
        self.alpha=alpha
        # class labels
        self.classes=None
        # priors P(y=c) for each class c
        self.priors={}
        # p(xj=1|y=c) for each feature xj and each class c
        self.probs={}

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier.
        
        Parameters:
        - X: Training data, shape (n_samples, n_features)
        - y: Target values, shape (n_samples,)
        """
        self.classes=np.unique(y)
        for c in self.classes:
            X_c=X[y==c]
            self.priors[c]= (X_c.shape[0] + self.alpha)/(X.shape[0]+ self.alpha * len(self.classes))
            self.probs[c]=(np.sum(X_c,axis=0)+ self.alpha)/(X_c.shape[0]+self.alpha*2)
        

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
                prob=np.log(self.priors[c])+ np.sum(x * np.log(self.probs[c]) + (1 - x) * np.log(1 - self.probs[c]))
                if prob>best_prob:
                    best_prob=prob
                    pred=c
            preds.append(pred)
        return np.array(preds)
