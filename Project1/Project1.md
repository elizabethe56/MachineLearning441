# Project 1


## What is Regression?

Regression is a technique of predicting an output, or dependent variable, given one or more inputs, or independent variables.

Mathematically, let $X$ be the inputs and $y$ be the outputs. Then, the regression equation can be shown as $$\mathbb{E}(y|X) = f(X).$$

Taking the expectation of the outputs demonstrates a minimization of the **mean squared error (MSE)**: the squared difference between the data points and the estimator function. By minimizing this, we can reasonably be sure that we have a good estimator function, based on the data we have.

***Linear* regression** specifies the use of a straight line for the regression, rather than a curve.


So, let $X=\{x_1,x_2,...,x_n\}$ be the inputs, $ B =\{\beta_1,\beta_2,...,\beta_n\}$ be the coefficients, and $y$ be the outputs.  Then the linear regression model is $$y=X \cdot B=x_1 \cdot \beta_1 + x_2 \cdot \beta_2 + ... + x_n \cdot \beta_n.$$

The only problem with this equation is the noise.  It is rare that the inputs and outputs are perfectly correlated, so it is safe to assume there is some noise variable for each point, randomly distributed on the Gaussian curve, leaving us with the equation $$y=X \cdot B + \sigma \epsilon$$

To solve for the coefficients, there are several matrix manipulations we must make to preserve the shape of the input features.  

First, we multiply by the transpose of $X$ to make $X^TX$ a square, then  multiply by the inverse of $X^TX$ (assuming $X^TX$ is invertible).  

$$\begin{align} X^Ty &= X^TX \cdot B + \sigma X^T \epsilon \\
(X^TX)^{-1}(X^Ty) &= (X^TX)^{-1}(X^TX) \cdot B + \sigma (X^TX)^{-1}(X^T \epsilon) \end{align}$$

Since a matrix times its own inverse is equal to the identity matrix, we then solve for the coefficient vector, $B$.  

$$\begin{align} 
(X^TX)^{-1}(X^Ty) &= B + \sigma (X^TX)^{-1}(X^T \epsilon) \\
B &= (X^TX)^{-1}(X^Ty) - \sigma (X^TX)^{-1}(X^T \epsilon) \\
\end{align}$$

The expectation (mean) of the noise variable is zero, so taking the expectation of the coefficients (evaluating MSE) eliminates the noise variable, leaving us with the expected value of the coefficients.

$$\bar{B} = (X^TX)^{-1}(X^Ty)$$

If we insert these estimated coefficients back into the equation, we get our predicted outputs:

$$\hat{y} = X \cdot (X^TX)^{-1}(X^Ty)$$

The last important piece of this puzzle, is the weights.  Certain features are going to correlate more with the output and the weights are how correlated features are emphasized and uncorrelated or less-correlated features are deemphasized in the estimation process.  The weights ($W$) are associated with the inputs, so our final equation looks like this:

$$\hat{y} = X \cdot (X^TWX)^{-1}(X^TWy)$$

## Locally Weighted Regression

Locally weighted regression is a method of executing linear regression on data with nonlinear trends and/or associations.  However, rather than using the whole data set for the regression, locally weighted regression evaluates the regression at multiple points of interest, with the weights based on the data points' distance from the point of interest.

For example, in the image below, the point of interest is the vertical line, and since the points nearest the point of interest have a negative linear trend, the line of regression has a negative slope.  The points farther away from the point of interest demonstrate a nonlinear trend, but since they are farther away, their weights are smaller (if not zero) and have little to no influence on the line of regression.

![LWR_ex.png](https://miro.medium.com/max/1400/1*H3QS05Q1GJtY-tiBL00iug.webp)

### Kernels

Locally weighted regression typically relies on a specified kernel to calculate the weights.  Some common ones are Gaussian, Tricubic, Epanechnikov, and Quartic.  These kernels determine the shape of the curve that calculates the weights, placing more or less significance on the surrounding points depending on the kernel.

Here are some examples of those kernels:


```python
# Define each kernel's function
def Gaussian(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*d**2))

def Tricubic(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

def Epanechnikov(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 

def Quartic(x):
  if len(x.shape)==1:
    d = np.abs(x)
  else:
    d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2) 
```


```python
# Define outputs for each kernel
x = np.linspace(-1,1,201)
y_gaussian = Gaussian(x)
y_tricubic = Tricubic(x)
y_epanechnikov = Epanechnikov(x)
y_quartic = Quartic(x)
```


```python
# Plot each kernel
fig, axs = plt.subplots(2,2)

axs[0,0].set_title('Gaussian Kernel')
axs[0,0].plot(x, y_gaussian)
axs[0,1].set_title('Tricubic Kernel')
axs[0,1].plot(x, y_tricubic)
axs[1,0].set_title('Epanechnikov Kernel')
axs[1,0].plot(x, y_epanechnikov)
axs[1,1].set_title('Quartic Kernel')
axs[1,1].plot(x, y_quartic)

for ax in axs.flat:
    ax.label_outer()

plt.show()
```


    
![png](output_7_0.png)
    


## Examples


```python
# calculate the difference with accomodations for the bandwidth, tau, 
# then sends it to one of the kernel function from above
def kernel_function(xi, x0, kern, tau): 
    return kern((xi - x0)/(2*tau))
```


```python
# creates the matrix of weights with x_new as the points of interest
def weights_matrix(x, x_new, kern, tau):
  if np.isscalar(x_new):
    return kernel_function(x, x_new, kern, tau)
  else:
    n = len(x_new)
    return np.array([kernel_function(x, x_new[i], kern, tau) for i in range(n)])
```


```python
class Lowess:
    # Initialize the kernel and tau in the initial call for Lowess
    def __init__(self, kernel = Gaussian, tau=0.05):
        self.kernel = kernel
        self.tau = tau
    
    # establish the training data
    def fit(self, x, y):
        kernel = self.kernel
        tau = self.tau
        self.xtrain_ = x
        self.yhat_ = y

    # the meat of the code; establishes the weights and evaluates (and returns) the predicted output for x_new
    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        lm = LinearRegression()

        w = weights_matrix(x,x_new,self.kernel,self.tau)

        # np.diag(w) creates a square matrix with the weights along the diagonal
        # lm.fit(...) evaluates the regression at each point of interest (x_new) with the weights established in w

        if np.isscalar(x_new):
          lm.fit(np.diag(w).dot(x.reshape(-1,1)),np.diag(w).dot(y.reshape(-1,1)))
          yest_test = lm.predict([[x_new]])[0][0]
        elif len(x.shape)==1:
          n = len(x_new)
          yest_test = np.zeros(n)
          #Looping through all x-points
          for i in range(n):
            lm.fit(np.diag(w[i,:]).dot(x.reshape(-1,1)),np.diag(w[i,:]).dot(y.reshape(-1,1)))
            yest_test[i] = lm.predict(x_new[i].reshape(-1,1))
        else:
          n = len(x_new)
          yest_test = np.zeros(n)
          #Looping through all x-points
          for i in range(n):
            lm.fit(np.diag(w[i,:]).dot(x),np.diag(w[i,:]).dot(y.reshape(-1,1)))
            yest_test[i] = lm.predict(x_new[i].reshape(1,-1))
        return yest_test
```

### Simulated data


```python
# Create sine function 
x = np.linspace(0,1,100)
noise = np.random.normal(loc = 0, scale = .25, size = 100)
y = np.sin(x * 1.5 * np.pi ) 
y_noise = y + noise

# Evaluate Lowess predictions
model_lw = Lowess(kernel = Gaussian, tau = 0.05)
model_lw.fit(x, y_noise)
y_lw = model_lw.predict(x)

# Evaluate Random Forest predictions
model_rf = RandomForestRegressor(n_estimators=60, max_depth=5)
model_rf.fit(x.reshape(-1,1), y_noise)
y_rf = model_rf.predict(x.reshape(-1,1))

plt.scatter(x, y_noise, facecolors = 'none', edgecolor = 'darkblue', label = 'f(x) + noise')
plt.plot(x, y, color = 'darkblue', label = 'Target function f(x)')
plt.plot(x, y_lw, color = 'red', label = 'Lowess Model')
plt.plot(x, y_rf, color = 'green', label = 'Random Forest Model')
plt.legend()
plt.title('Noisy sine function')

plt.show()
```


    
![png](output_13_0.png)
    


### Real Data


```python
# import the cars data
data = pd.read_csv('cars.csv')

# Establish MPG as the output and ENG and WGT as inputs
cars_x = data.loc[:,'ENG':'WGT'].values
cars_y = data['MPG'].values
```


```python
# use K-Fold cross validation to get a clearer picture of the true MSE
kf = KFold(n_splits=10,shuffle=True,random_state=123)

# must use a scaler for multidimensional data
scale = StandardScaler()

mse_test_lowess = []
mse_test_rf = []

for idxtrain, idxtest in kf.split(cars_x):
  xtrain = scale.fit_transform(cars_x[idxtrain])
  xtest = scale.transform(cars_x[idxtest])
  ytrain = cars_y[idxtrain]
  ytest = cars_y[idxtest]

  # Evaluate Lowess, calculate and store the MSE
  model_lw = Lowess(kernel=Gaussian, tau=0.085)
  model_lw.fit(xtrain,ytrain)
  mse_test_lowess.append(mse(ytest,model_lw.predict(xtest)))

  # Evaluate Random Forest, calculate and store the MSE
  model_rf = RandomForestRegressor(n_estimators=200,max_depth=8)
  model_rf.fit(xtrain,ytrain)
  mse_test_rf.append(mse(ytest,model_rf.predict(xtest)))

print('The validated MSE for Lowess is : '+str(np.mean(mse_test_lowess)))
print('The validated MSE for Random Forest is : '+str(np.mean(mse_test_rf)))
```

    The validated MSE for Lowess is : 29.68400537458526
    The validated MSE for Random Forest is : 18.66526320676156

