# Machine Learning Algorithms From Scratch
## [Linear Regression](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/tree/main/Linear%20Regression)
Prediction function: 
$$y=Xw+\epsilon,$$
where $y=[y_1,y_2,...,y_n] \in \mathbb{R}^n$ is the output value, $X=[X_1, X_2,...,X_n]$ is a nxd input matrix, and $w=[w_1, w_2, w_d] \in \mathbb{R}^d$ is the vector of parameters.

Object function: 
$$RSS(w)=\|\|Xw-y\|\|^{2}_{2}$$

### [Linear Regression - Closed-form Solution](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/blob/main/Linear%20Regression/linReg.py)
$$w^*=(X^TX)^{-1}X^Ty$$

### [Linear Regression - Gradient Descent](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/blob/main/Linear%20Regression/linRegGD.py)
$$w_{t+1}=w_t - \alpha * \frac{\partial RSS(w)}{\partial w},$$ where $\alpha$ is the learning rate of gradient descent

## [Polynomial Regression](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/blob/main/Linear%20Regression/polyReg.py)
Prediction function:
$$y=\phi w+\epsilon$$
$$\phi=[\phi(x_1), \phi(x_2),...,\phi(x_n)]^T$$
$$\phi(x_i)=[x_i,x_i^2,...,x_i^k],$$
where $k$ is the degree of polynomials, $\phi(x_i)$ is the polynomial basis function, and $\phi$ is a nxk matrix of the polynomial features

Object function: 
$$RSS(w)=\|\|\phi w-y\|\|^{2}_{2}$$

**Closed-form Solution:** 
$$w^*=(\phi^T\phi)^{-1}\phi^Ty$$

## [Lasso Regression](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/blob/main/Linear%20Regression/lassoReg.py)
Prediction function: 
$$y=Xw+\epsilon$$
Object function: 
$$RSS(w,b)=\|\|Xw+b-y\|\|^{2}_{2}+\lambda\|w\|,$$ where $\lambda$ is the regularization parameter

**Gradient Descent Solution:**
$$w_{t+1}=w_t - \alpha * \frac{\partial RSS(w,b)}{\partial w}$$

$$b_{t+1}=b_t - \alpha * \frac{\partial RSS(w,b)}{\partial b}$$ 

where $\alpha$ is the learning rate of gradient descent
$$\frac{\partial RSS(w,b)}{\partial w} = X^T(Xw+b-y)+\lambda*sign(w)$$
$$\frac{\partial RSS(w,b)}{\partial b} = \|\|Xw+b-y\|\|$$

**Note:** In Lasso Regression, we should exclude the bias term from regularization.

## [Ridge Regression](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/blob/main/Linear%20Regression/ridgeReg.py)
Prediction function: 
$$y=Xw+\epsilon$$
Object function: 

$$RSS(\mathbf{w}, b) = \|\|Xw + b - y\|\|_2^2 + \lambda \|\|w\|\|_2^2$$

where $\lambda$ is the regularization parameter

**Closed Form Solution:** $$w^*=X^T(Xw-y)+2I_d\lambda,$$
where $I \in \mathbb{R}^d$ is an identity matrix whose [0,0] entry is 0.

**Note:** In Ridge Regression, we should exclude the bias term from regularization.

## [Naive Bayes](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/tree/main/Naive%20Bayes)

**Problem Setting**

* A data set with n samples ${(x_1,y_1), (x_2,y_2),...,(x_3,y_3)}$
* each $x_i$ is a feature vector: $x_1=[x_{i1}, x_{i2},...,x_{id}]$ where $d$ is the number of features
* $y_i$ is the class label for $x_i$ and takes values from a set of classes $C$

**Bayes Theorem**

The posteriao probability for a class $c$ givne a feature vector $x$ is:
$$P(y=c|x)=\frac{P(x|y=c)P(y=c)}{P(x)}$$
$$posterior \varpropto likelihood * prior$$
where:
* $P(y=c|x)$ is the **likelihood**, which is the probability of observing the feature vector $x$ given that this sample is of class $c$.
* $P(y=c)$ is the **pripor**, whcih is the porbability of any sample being of class $c$ without observing it.
* $P(x)$ is the evidence, which is probability of observing the feature vector $x$

**Naive Assumption**

The "naive" in Naive Bayes comes from the assumption that **each feature in the dataset is independent of all other features, given the class label**. This allows us to simplify the likelihood as:
$$P(x | y=c) = \prod_{j=1}^{d} P(x_j | y=c)$$

**Classification**

To classify a new sample $x$, we compute the posterior probability for each class c and choose the class with the highest posterior probability. The evidence $P(x)$ is the same for all classes, so it can be ignored for this purpose:
$$\hat{y} = \arg\max_{c \in C} P(y=c) \times \prod_{j=1}^{d} P(x_j | y=c) = \arg\max_{c \in C}\log P(y=c) + \sum_{j=1}^{d} \log P(x_j | y=c)
$$
where $\hat{y}$ is the predicted label for $x$.

### [Gaussian Naive Bayes](https://github.com/nclw1118/Machine-Learning-Algorithms-From-Scratch/blob/main/Naive%20Bayes/GaussianNaiveBayes.py)
The likelihood of a feature given a class is computed using the Gaussian probability density function. Primarily used for continuous features that can be assumed to have a gaussian distribution. 

**Gaussian Assumption**

For Gaussian Naive Bayes, we assume that the continuous values associated with each class are distributed according to a Gaussian distribution. For each feature $j$ and class $c$, the likelihood of a value $x_{j}$ given the class $c$ is:
$$P(x_j | y=c) = \frac{1}{\sqrt{2\pi\sigma^2_{jc}}} \exp\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma^2_{jc}}\right)$$
where:
* $\mu_{jc}$ is the mean of feature $j$ for samples from class $c$
* $\sigma_{jc}^{2}$ is the variance of feature $j$ for samples from class $c$

### Bernoulli Naive Bayes (with Laplace Smoothing)

**Bernoulli Distribution**

The probability mass function of the Bernoulli distribution is:
$$P(X=k) = p^k (1-p)^{1-k}$$
where $k$ can be either 0 or 1, and $p$ is the probability of success(i.e.$X=1$)

**Laplace Smoothing**

Laplace smoothing in Naive Bayes prevents zero probabilities for unseen feature-class combinations, ensuring the model remains applicable to new data by assigning small non-zero probabilities to unobserved events.

Laplace smoothing (or add-one smoothing) when estimating probabilities from frequency counts is:
$$P(x) = \frac{\text{count}(x) + \alpha}{N + \alpha k}$$
where:
* $\text{count}(x)$ is the number of times event x occurs in the data.
* $N$ is the total number of events.
* $k$ is the number of possible distinct events.
* $\alpha$ is the smoothing parameter

With Laplace Smoothing, the smoothed prior probability $P(y_i)=c$ for a class $c$ is:
$$P(y_i = c) = \frac{\text{count}(y_i = c) + \alpha}{N + \alpha k}$$

**Bernoulli Assumption**

For Bernoulli Naive Bayes, we assume that the binary values associated with each class are distributed according to a Bernoulli distribution. For each feature $j$ and class $c$, the likelihood of a value $x_{j}$ given the class $c$ is:
$$P(x_j | y=c) = p_{jc}^{x_j}(1-p_{jc})^{(1-{x_j})}=P(x_j=1|y=c)^{x_j}(1-P(x_j=1|y=c))^{(1-{x_j})}$$
where $p_{jc}=P(x_j=1|y=c)$ is the probability of each feature being 1 (or present) given a class. For each feature $j$ and class $c$:
$$P(x_j = 1 | y=c) = \frac{\sum_{i=1}^{n} I(x_{ij} = 1 \land y_i = c) + \alpha} {\sum_{i=1}^{n} I(y_i = c) + 2\alpha}$$
where:
* $I(\cdot)$ is the indicator function that returns 1 if the condition inside is true and 0 otherwise.
* $\alpha$  is the Laplace smoothing parameter (typically set to 1 for one-unit smoothing).
