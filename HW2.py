#***********************
# Title: HW2
# Purpose: Grid Search
# Author: Dan Crouthamel, Fabio Savorgnan
# Date: January 2021
#***********************
# import numpy as np
# from sklearn.metrics import accuracy_score # other metrics too pls!
# from sklearn.ensemble import RandomForestClassifier # more!
# from sklearn.model_selection import KFold

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

# Comment out provided code
# M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
# L = np.ones(M.shape[0])
# n_folds = 5

# data = (M, L, n_folds)

# def run(a_clf, data, clf_hyper={}):
#   M, L, n_folds = data # unpack data container
#   kf = KFold(n_splits=n_folds) # Establish the cross validation
#   ret = {} # classic explication of results

#   for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
#     clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist
#     clf.fit(M[train_index], L[train_index])
#     pred = clf.predict(M[test_index])
#     ret[ids]= {'clf': clf,
#                'train_index': train_index,
#                'test_index': test_index,
#                'accuracy': accuracy_score(L[test_index], pred)}
#   return ret

# results = run(RandomForestClassifier, data, clf_hyper={})

#########################################################
################### Begin Solution ######################
#########################################################

# The code above was provided to start with. I've commented it out.
# I created a class and then some test cases afterwards.
# My run implementation is a bit different, and I'm going to use cross_val_score which includes Kfolds
# A nice exercise in working with different data structures, dictionaries within dictionaries, etc.
# Note, there is no error checking and no thought as to whether or not the modesl/params 
# make sense for a given dataset.

import numpy as np
import itertools as it
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
import matplotlib.pyplot as plt

class GridSearch(object):
    """
    Grid search for multiple models and metrics.
    Would be useful to complete this, but really don't have time ;0
    ...

    Attributes
    ----------
    xxx

    Methods
    -------
    xxx
    """

    def __init__(self, X, y, models_params={}, metrics=[]):
        self.M = X
        self.L = y
        self.models_params = models_params
        self.metrics = metrics
        self.grid_results = []
        self.best_scores = []  # data structure used to print out scores
        self.best_metrics = {} # data structure used to plot/visualize

        # Initialize our best scores dictionary data structure
        # 
        # 'clf' = Classifier Object
        # 'best_scores' = {'accuracy': 0, 'recall': 0, etc} -> dependent upon metric input
        # 'best_params' = {'accuracy': {}, 'recall': {}} -> initialize to empty dictionary
        # The dictionary structures will later contain the actually parameters that were used to obtain the best score
        for x in models_params:
            self.best_scores.append({'clf': x, 
                                     'best_scores': dict.fromkeys(metrics,0),
                                     'best_params': {k: {} for k in metrics}
                                    })

        # Create container based on metric
        # This will be a list of best scores for a given metric, in the order of input classifiers
        #
        # Ahh - don't do this!
        # List are mutable and updating one key updated all of them
        # self.best_metrics = dict.fromkeys(metrics,[])
        self.best_metrics = {k: [] for k in metrics}

    def grid_search(self):
        for model, params in self.models_params.items():
            print("")
            print("Evaluating ...", model.__name__, "...")
            #
            # Keys = hyperparmeters, e.g. 'solver'
            # Values = e.g., 'newton-cg', 'lbfgs'
            # We have multiple values for a key (hyperparameter) that we wish to test
            keys, values = zip(*params.items())

            #It's worth mentioning how this works, if it's new to you
            # Starting on the far right:
            #   it.product(*values) = we unupack our values (a tuple of lists, with each list being a list
            #   of possible values for a given hyperparameter) and create an iterable of possible combinations 
            #   that we can loop over.
            #   Value that is returned is a tuple containing one value for each hyperparmater
            #   So we can zip that with our keys and create a dictionary of every possible combination
            #   of hyperparameters and values
            paramsToPassToRun = [dict(zip(keys, value)) for value in it.product(*values)]
            
            #Can probably use a list comprehension here
            for runParams in paramsToPassToRun:
                results = self.__run(model, runParams, self.metrics)
                self.grid_results.append(results)
        
        # And this too, a comprehension it could be
        #Update our metrics structure
        for model in self.best_scores:
            for metric in model['best_scores']:
                self.best_metrics[metric].append(model['best_scores'][metric])

        return self.best_scores

    # Our run implementation is a bit different than what was assigned.
    # I changed it to use cross_val_score with cv=5, same 5 K-folds above
    def __run(self, a_clf, clf_hyper={}, clf_metrics={}):
        clf = a_clf(**clf_hyper) # unpack parameters into clf as they exist

        ret = {}

        for metric in clf_metrics:
            scores = cross_val_score(clf, X=self.M, y=self.L, cv=StratifiedKFold(n_splits=5), scoring=metric, n_jobs=-1)
            ret.update({'clf': clf,
                        'clf_params': clf_hyper,
                        metric: scores})

            # Update our collection of best mean scores
            mean_score = scores.mean()
            for model in self.best_scores:
                if model['clf'] == a_clf and model['best_scores'][metric] < mean_score:
                    model['best_scores'][metric] = mean_score
                    model['best_params'][metric] = clf_hyper
        
        return ret

    # Quick text dump, nothing pretty
    def print_scores(self):
        for model in self.best_scores:
            print("")
            print(model['clf'].__name__)

            for metric in model['best_scores']:
                print(metric, "{:.2f}".format(model['best_scores'][metric]), "-", model['best_params'][metric])

    # Quick and ugly plot
    def plot_metric_scores(self):
        clfs_list = [clf.__name__ for clf in model_params]
        num_plots = len(gs.best_metrics)
        index = 0

        plt.figure(figsize=(16,8))
        for metric in gs.best_metrics:
            index = index + 1
            plt.subplot(num_plots, 2, index)
            #plt.ylim(0, 1)
            plt.title(metric)
            plt.plot(clfs_list, gs.best_metrics[metric])
        plt.tight_layout()
        #plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=1)
        plt.show()

## Let's test our class!

# Define some data to work with
# Limit to Iris-versicolor and Iris-virginica, so just a binary classification problem
iris = datasets.load_iris()
X = iris.data[50:]
y = iris.target[50:]

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

# Setup the models and params we want to test
model_params = {
    RandomForestClassifier: { 
        #"n_estimators" : [100, 200, 500, 1000],
        "n_estimators" : [50, 100],
        "max_features" : ["auto", "sqrt", "log2"],
        "bootstrap": [True],
        "criterion": ['gini', 'entropy'],
        "oob_score": [True, False]
        },
    KNeighborsClassifier: {
        'n_neighbors': np.arange(3, 15),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },
    LogisticRegression: {
        #'solver': ['newton-cg', 'sag', 'lbfgs'],
        'solver': ['newton-cg', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial'],
        'max_iter': [100, 1000]
        }
}

# Setup the metrics we want to test
metrics = ['accuracy', 'roc_auc', 'recall', 'precision']

gs = GridSearch(X=X_std, y=y, models_params=model_params, metrics=metrics)
best_scores = gs.grid_search()
gs.print_scores()
gs.plot_metric_scores()

# Let's try a different data set!
cancer = datasets.load_breast_cancer()
X = cancer['data']
y = cancer['target']

# Let's scale with RobustScaler
# MLPClassifier throws a bunch of warnings, but then again
# NO THOUGHT was put into if these models, hyperparams and metrics even make sense for our data set!
# It's been a playful exercise in python data structures
robust_scaler = RobustScaler()
robust_scaler.fit(X)
X_std = robust_scaler.transform(X)

# Let's use different set of model/params
# Note, I'm passing random_state below because I'm going to compare later the GridSearchCV function
model_params = {
    RandomForestClassifier: { 
        "n_estimators" : [50, 100],
        "max_features" : ["auto", "sqrt", "log2"],
        "bootstrap": [True],
        "criterion": ['gini', 'entropy'],
        "oob_score": [True, False],
        "random_state": [0]
        },
    MLPClassifier: {
        'hidden_layer_sizes': (3,3),
        'alpha': [.0001, .001, .01],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [1000, 2000]
        },
    LogisticRegression: {
        'solver': ['newton-cg', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial'],
        'max_iter': [100, 1000]
        }
}

gs = GridSearch(X=X_std, y=y, models_params=model_params, metrics=metrics)
best_scores = gs.grid_search()
gs.print_scores()
gs.plot_metric_scores()

# Compare with the GridSearchCV from Sklearn
# Note, I went back and added StratifiedKfold and random_state
# to see if the results matches.
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state=0)
model_params= {
        "n_estimators" : [50, 100],
        "max_features" : ["auto", "sqrt", "log2"],
        "bootstrap": [True],
        "criterion": ['gini', 'entropy'],
        "oob_score": [True, False]
        }
cv = StratifiedKFold(n_splits=5)

# RandomForest and Precision
clf = GridSearchCV(estimator=model, param_grid=model_params, cv=cv, scoring="precision")
clf.fit(X_std, y)
print("Best: %f using %s" % (clf.best_score_, clf.best_params_))

# RandomForest and Roc_auc
clf = GridSearchCV(estimator=model, param_grid=model_params, cv=cv, scoring="roc_auc")
clf.fit(X_std, y)
print("Best: %f using %s" % (clf.best_score_, clf.best_params_))

# If you look at the output from GridSearchCV and compare to our own method,
# you'll see the results are the same!
