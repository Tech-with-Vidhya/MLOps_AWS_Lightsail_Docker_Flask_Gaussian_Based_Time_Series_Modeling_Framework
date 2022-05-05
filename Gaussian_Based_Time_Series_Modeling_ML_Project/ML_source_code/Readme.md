# Gaussain Processes Regression Time Series

Gaussian Processes are a generalization of the Gaussian probability distribution and can be used as the basis for sophisticated non-parametric machine learning algorithms for classification and regression.
Gaussian probability distribution functions summarize the distribution of random variables, whereas Gaussian processes summarize the properties of the functions, e.g. the parameters of the functions.
Gaussian processes can be used as a machine learning algorithm for classification predictive modelling
## Correlation vs AutoCorrelation

- Correlation is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship. In terms of the strength of relationship, the value of the correlation coefficient varies between +1 and -1.
- A value of ± 1 indicates a perfect degree of association between the two variables. As the correlation coefficient value goes towards 0, the relationship between the two variables will be weaker.
- Auto-correlation refers to the case when your errors are correlated with each other. In layman terms, if the current observation of your dependent variable is correlated with your past observations, you end up in the trap of auto-correlation. 

## Gaussian Kernel

Gaussian processes require specifying a kernel that controls how examples relate to each other; specifically, it defines the covariance function of the data.
The way that examples are grouped using the kernel controls how the model “perceives” the examples, given that it assumes that examples that are “close” to each other have the same class label.
Therefore, it is important to both test different kernel functions for the model and different configurations for sophisticated kernel functions.


## Time Series Basics

-   Chronological Data
- Cannot be shuffled
- Each row indicate specific time record
- Train – Test split happens chronologically
- Data is analyzed univariately (for given use case)
- Nature of the data represents if it can be predicted or not

## Code Description


    File Name : Engine.py
    File Description : Main class for starting different parts and processes of the lifecycle


    File Name : Gaussian_Stationary.py
    File Description : Code to train and visualize Gaussian process with Stationary data


    File Name : Gaussian_Trend.py
    File Description : Code to train and visualize Gaussian process with trend data



## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `Gaussian_Process.ipynb`

