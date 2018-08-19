
# coding: utf-8

# <h1>First Machine Learning Project </h1>
# <p>In this project I'll try to use simple ML algorithms to learn machine learning </p>
# <h4> Below are the main Steps or phases that I will go throguh in this project:</h4>
#     <li>Define Problem</li>
#     <li>Prepare Data</li>
#     <li>Explore Data</li>
#     <li>Evaluate Algorithms</li>
#     <li>Improve Results</li>
#     <li>Present Results</li>
# 
# 
# 
# .

# In[2]:


# Check Versions
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))



# Load libraries.
# Below are the libraries that we are going to use in this project. 
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset.
# Note that we are specifying the column names now which will help us later on. 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# <h4>In this step we are going to take a look at the data a few different ways:</h4>
# 
# <li>Dimensions of the dataset</li>
# <li>Peek at the data itself</li>
# <li>Statistical summary of all attributes</li>
# <li>Breakdown of the data by the class variable</li>

# Dimentions (rows, columns)
dataset.shape

# Peek at the data 
dataset.head(20)



# Statistical Summary 
dataset.describe()


# Class distribution 
print(dataset.groupby('class').size())


# <h3> Data Visualization </h3>

# <p>We are going to look at two types of plots:</p>
# 
# <li>Univariate plots to better understand each attribute</li>
# <li>Multivariate plots to better understand the relationships between attributes</li>

# <h3> Univariate plots </h3>


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# histograms 
# histogram is only used to plot the frequency of score occurrences in a continuous data set that has been divided into classes, 
# called bins.
dataset.hist()
plt.show()


# <h3> Multivariate plots </h3>


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# <h3>Evaluate Some Algorithms</h3>

# <h4>Here is what we are going to cover in this step:</h4>
# 
# <li>Separate out a validation dataset</li>
# <li>Set-up the test harness to use 10-fold cross validation</li>
# <li>Build 5 different models to predict species from flower measurements</li>
# <li>Select the best model</li>

# <h4>Create a Validation Dataset</h4>
# <p>We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset</p>


# Assign the dataset df to an array to split its values 
array = dataset.values

# Split-out validation dataset
array = dataset.values
X = array[:,0:4] # Input columns 
Y = array[:,4] # output column
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# <h3>Test Harness</h3>
# <p>We will use 10-fold cross validation to estimate accuracy.</p>
# 
# <p>This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.</p>

# <p>We are using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.</p>

# Test options and evaluation metric to be used in Cross-validation 
seed = 7
scoring = 'accuracy'


# <h3>Build Models</h3>

# We don’t know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.
# 
# Let’s evaluate 6 different algorithms:
# 
# <li>Logistic Regression (LR)</li>
# <li>Linear Discriminant Analysis (LDA)</li>
# <li>K-Nearest Neighbors (KNN).</li>
# <li>Classification and Regression Trees (CART).</li>
# <li>Gaussian Naive Bayes (NB).</li>
# <li>Support Vector Machines (SVM).</li>

# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.
# 

# Building our models
models = [] 
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn 
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# <h2>Make Predictions</h2>

# The SVM algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation set.
# 
# This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.
# 
# We can run the SVM model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.


# Make predictions on validation dataset
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation) # Predict testing dataset
print("KNN")
print('Accuracy Score: \n {0}'.format(accuracy_score(Y_validation, predictions)))
print('Confusion Matrix: \n {0}'.format(confusion_matrix(Y_validation, predictions)))
print('Classification Report: \n {0}'.format(classification_report(Y_validation, predictions)))


lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print('LR')
print('Accuracy Score: \n{0}'.format(accuracy_score(Y_validation, predictions)))
print('Confusion Matrix: \n {0}'.format(confusion_matrix(Y_validation, predictions)))
print('Classification Report: \n {0}'.format(classification_report(Y_validation, predictions)))

cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print('CART')
print('Accuracy Score: \n{0}'.format(accuracy_score(Y_validation, predictions)))
print('Confusion Matrix: \n {0}'.format(confusion_matrix(Y_validation, predictions)))
print('Classification Report: \n {0}'.format(classification_report(Y_validation, predictions)))

