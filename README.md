# Problem Title

Deep Learning in Astrophysics

# Description

The tentative goal of this project is to predict the redshift of the galaxies using the available photometric data. The photometric data is available for a very broad band, i.e. very few data points. The
neural network will train itself on the available spectroscopic data, and will predict the same for the photometric data. The project involves the use of Multitask Learning. It will 

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

In all the above cases, the predicted output was plotted against the available output as a scatter plot to give a visual idea of the accuracy of the data. The arithematic output was calculated by finding
the **Mean-Squared Error** in each case. Since the K-Fold Cross Validation leads to the different number of iterations, we get different values of MSE for each algorithm. The mean of these MSEs were
calculated and used as a metric for comparison.

## Deep Learning: Neural Network

### Objective

The primary objective of this module is to improve the accuracy of the already used classic machine learning algorithms by the use of multi-layered neural networks. The idea is to prepare an ideal Neural
Network to predict the redshift values from the available Photometric points. 

### Dataset used

The same [dataset](data) from the previous machine learning model was used in this case.

### Algorithm Implementation

Two major libraries used in Artificial Neural Networks are [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/). Both of them complement each other. After the cleaning of the dataset,
a random neural network was prepared using random number of layers and units in each layer. The **epoch** and **batch-size** was varied and the algorithm was tested which would provide the best fit. Once
these were fixed, the testing started for the optimum number of hidden layers and the units in these layers. The testing was completely manual and the output was verified using the plots and the MSE
metric.

## Multitask Learning

### Objective

The immediate objective is to prepare two neural network. One of which will train on the broad band photometric data and predict the photometric data for the narrow band. The other one will take input
the combination of the broad band and the narrow band spectrum as predicted by the previous.

### Paper method

The given paper which is supposed to be reproduced provides a single MTL network, which takes in broadband photometry and gives output both, the narrowband photometry as well as the broadband photo-z. The paper claims that the preeiction of narrowband photometry improves the prediction of photometric redshifts. 

### Dataset used

The [dataset](PAU_narrowband_data_full.fits) used is from PAU Survey. It has the data of approximately 6500 galaxies. It consists of 

1. Broadband fluxes from the following bands:
- u
- b
- v
- g
- r
- z
- ic
- j
- k

2. Narrowband fluxes from 455 nm to 855 nm

3. Spectroscopic redshifts.

### Algorithm implemented

The algorithm uses the [Pytorch](https://pytorch.org/) library to create two types of neural networks and compare their performances. The first one is a simple linear neural network, which can be presented as:
![Linear Neural Network](https://github.com/hrishabhsrivastava/GRI2022/blob/main/Model/Linear%20Network.png)
