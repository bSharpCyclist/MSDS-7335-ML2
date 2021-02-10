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
# A nice exerise in working with different data structures, dictionaries within dictionaries, etc.

import numpy as np
import itertools as it
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt

# Describe class inputs

class GridSearch(object):

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
            keys, values = zip(*params.items())
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
    # I changed it to use cross_val_score
    def __run(self, a_clf, clf_hyper={}, clf_metrics={}):
        clf = a_clf(**clf_hyper) # unpack parameters into clf as they exist

        ret = {}

        for metric in clf_metrics:
            scores = cross_val_score(clf, X=self.M, y=self.L, cv=5, scoring=metric, n_jobs=-1)
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
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
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

scaler.fit(X)
X_std = scaler.transform(X)

gs = GridSearch(X=X_std, y=y, models_params=model_params, metrics=metrics)
best_scores = gs.grid_search()
gs.print_scores()
gs.plot_metric_scores()

# Compare with the grid search from Sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
lr= LogisticRegression()
model_params= {
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
        }
#make CV spit 80/20 object
num_cv_iterations = 3
cv_object = ShuffleSplit(n_splits=num_cv_iterations,
                         test_size  = 0.2)
                         
clf = GridSearchCV(lr, model_params, scoring="roc_auc", cv=cv_object)
clf.fit(X_std, y)

print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
