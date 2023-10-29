# Machine Learning Algorithms From Scratch
## Linear Regression
Prediction function: 
$$y=Xw+\epsilon,$$
where $y=[y_1,y_2,...,y_n] \in \mathbb{R}^n$ is the output value, $X=[X_1, X_2,...,X_n]$ is a nxd input matrix, and $w=[w_1, w_2, w_d] \in \mathbb{R}^d$ is the vector of parameters.

Object function: 
$$RSS(w)=\|\|Xw-y\|\|^{2}_{2}$$

### Linear Regression - Closed-form Solution
$$w^*=(X^TX)^{-1}X^Ty$$

### Linear Regression - Gradient Descent
$$w_{t+1}=w_t - \alpha * \frac{\partial RSS(w)}{\partial w},$$ where $\alpha$ is the learning rate of gradient descent

## Polynomial Regression
Prediction function:
$$y=\phi w+\epsilon$$
$$\phi=[\phi(x_1), \phi(x_2),...,\phi(x_n)]^T$$
$$\phi(x_i)=[x_i,x_i^2,...,x_i^k],$$
where $k$ is the degree of polynomials, $\phi(x_i)$ is the polynomial basis function, and $\phi$ is a nxk matrix of the polynomial features

Object function: 
$$RSS(w)=\|\|\phi w-y\|\|^{2}_{2}$$

**Closed-form Solution:** 
$$w^*=(\phi^T\phi)^{-1}\phi^Ty$$

## Lasso Regression
Prediction function: 
$$y=Xw+\epsilon$$
Object function: 
$$RSS(w,b)=\|\|Xw+b-y\|\|^{2}_{2}+\lambda\|w\|,$$ where $\lambda$ is the regularization parameter

**Gradient Descent Solution:**
$$w_{t+1}=w_t - \alpha * \frac{\partial RSS(w,b)}{\partial w}$$

$$b_{t+1}=b_t - \alpha * \frac{\partial RSS(w,b)}{\partial b}$$ 
where $\alpha$ is the learning rate of gradient descent
$$\frac{\partial RSS(w,b)}{\partial w} = X^T(Xw+b-y)+\lambda*sign(w)$$
$$\frac{\partial RSS(w,b)}{\partial b} = \|Xw+b-y\|$$
