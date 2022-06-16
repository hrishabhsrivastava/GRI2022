# Problem Title

Deep Learning in Astrophysics

# Description

The tentative goal of this project is to predict the redshift of the galaxies using the available photometric data. The photometric data is available for a
very narrow band. The neural network will train itself on the available spectroscopic data, and will predict the same for the photometric data. 

# Learning Series

## Machine Learning

### Data Cleaning

The project started with importing the data matrix available in the [data](data) file. It was found that the photometric data for some of the galaxies at some specific wavelengths were inconsistent with
the expected values. This usually happens when the galaxy was not detected at that particular wavelength. It was labelled with a value of -99.0 which isn't a physically possible value. Similarly, some of
the redshift values were 0, which again is not physically possible. The data matrix was cleared was cleared with all those elements, leaving us with the data of 2043 galaxies.

### Splitting the matrix

The next task in the machine learning approach was to separate the test data from the train data. [K-fold Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) was used for the same. The data was splitted into 5 parts, with the first 4 parts used for training with the last part used for testing. In order to improve training of the model, the splitted
data matrix was iterated such that all the subgroups got a chance to be testing data once.

### Algorithm Implementation

The initial project started with a basic learning of all the basic ML algorithms. It covered the following:

1. [Gradient Boosted Regression Trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html)
2. [Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
3. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
4. [Adaboost Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
5. [K-Nearest Neighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
6. [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
7. [Support Vector Machine - RBF](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
8. [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

### Output and Comparison

In all the above cases, the predicted output was plotted against the available output as a scatter plot to give a visual idea of the accuracy of the data.  
