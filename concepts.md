

# Supervised learning

## Linear regression

- assume targets are a linear combination of features
- minimize ordinary least squares using gradient descent to calculate the parameters
- the normal equation also provides a closed-form solution for the parameters

Hypothesis $h(x) = \sum_{i=0}^{n} \theta_i x_i = \theta^T x$

- often use the convention $x_0 = 1$ which is an intercept term

Cost function $J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2$

Algorithm looks something like $\theta_j := \theta_j - \alpha \sum_{i=1}^{m} \frac{\partial}{\partial \theta_j} J(\theta) = \theta_j + \alpha \sum_{i=1}^{m} \left( y^{(i)} - h_{\theta}(x^{(i)}) \right) x_j^{(i)}$

- above is batch gradient descent, can also update by looking at only a single data point at a time and iterating through each point

For linear regression, solving the normal equation $\theta = (X^T X)^{-1} X^T \vec{y}$ also analytically gives the optimal parameters

## Probabilistic interpretation

- assume targets are linear function of the features plus a normally distributed error 
- define the likelihood as the probability of seeing the targets given the inputs
- maximize the log likelihood (since math is easier and log is monotonic function) to calculate the parameters
- mathematically, this is equivalent to minimizing ordinary least squares

Assume $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$ such targets are related to the inputs plus some error $\epsilon$

Assume $\epsilon$ is Gaussian with mean $0$ and stdev $\sigma^2$

This implies $y^{(i)}$ is also Gaussian $y^{(i)} \mid x^{(i)} ; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$

Now look at the likelihood $L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} \mid X; \theta)$

So the likelihood function is: 
$L(\theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta) \\
          = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)
$

Maximizing log likelihood $\log L(\theta)$ is equivalent to minimizing the cost function $J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2$

## Locally weighted linear regression

- introduce weights to the linear regression model so that data closest to the query point are most important

Instead of modeling to minimize $\sum_{i} \left( y^{(i)} - \theta^T x^{(i)} \right)^2$ instead minimize $\sum_{i} w^{(i)} \left( y^{(i)} - \theta^T x^{(i)} \right)^2$

Typically choose $w^{(i)} = \exp \left( -\frac{(x^{(i)} - x)^2}{2\tau^2} \right)$

Note here $x$ is the query point, so the calculation much be repeated for every query point needed (i.e. it does not produce an $h(x)$ for all query points like regular linear regression does)



# Classification and logistic regression

## Logistic regression

- apply sigmoid to linear regression
- model prediction based on $p(y \mid x; \theta) = \left( h_{\theta}(x) \right)^{y} \left( 1 - h_{\theta}(x) \right)^{1-y}$; e.g. $\theta^T x > 0$ predicts $y=1$
- update rule looks similar to linear regression although with different function

Apply sigmoid function $g(z) = \frac{1}{1 + e^{-z}}$ to linear regression hence $h_{\theta}(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$

Then model $p(y \mid x; \theta) = \left( h_{\theta}(x) \right)^{y} \left( 1 - h_{\theta}(x) \right)^{1-y}$

Likelihood is then $L(\theta) = \prod_{i=1}^{m} \left( h_{\theta}(x^{(i)}) \right)^{y^{(i)}} \left( 1 - h_{\theta}(x^{(i)}) \right)^{1 - y^{(i)}}$

By maximizing log likelihood arrive at update rule $\theta_{j} := \theta_{j} + \alpha \left( y^{(i)} - h_{\theta}(x^{(i)}) \right) x_{j}^{(i)}$

## Perceptron learning algorithm








xxx

# Linear algebra

- gradient $\nabla f$(x) for $f: R^n \rightarrow R$
- Hessian $\nabla^2 f$(x) for $f: R^n \rightarrow R$
- positive definite ($A \succ 0$) and positive semi-definite ($A \succeq 0$)
- null space and rank of matrices
- eigenvectors, eigenvalues, and the spectral theorem
- rank-nullity theorem
- important representations: $x^TAx$


# a

- batch gradient descent, mini-batch gradient descent, stochastic gradient descent
- normal equation for linear regression
- parametric learning
- logistic regression
- exponential families
- generalized linear regression
