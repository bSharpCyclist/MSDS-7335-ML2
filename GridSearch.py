#***********************
# Title: HW2
# Purpose: Grid Search
# Author: Dan Crouthamel, Fabio Savorgnan
# Date: January 2021
#***********************
import numpy as np
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from itertools import *

##### These are the assignment questions
# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function
#####

### Let's use the Iris data set, something simple to load and use.
#M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
#L = np.ones(M.shape[0])

# This would use just 2 features
#M = iris.data[:, [2, 3]]

# Limit to Iris-versicolor and Iris-virginica, so just a binary classification problem
iris = datasets.load_iris()
M = iris.data[50:]
L = iris.target[50:]

# We typically scale data. It will help the MLP classifier.
# Many algorithms require some sort of feature scaling for optimal performance
scaler = StandardScaler()
scaler.fit(M)
M_std = scaler.transform(M)

n_folds = 5
data = (M_std, L, n_folds)

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data container
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

# will return a dictionary of the clf, the params, and the scores of all the metrics
def run2(a_clf, data, clf_hyper={}, clf_metrics={}):
    clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist
    M, L, n_folds = data
    
    # scores = {}
    # scores = cross_val_score(clf, X=M, y=L, cv=5, scoring='accuracy', n_jobs=-1)
    # return scores

    ret = {}

    for metric in clf_metrics:
        scores = cross_val_score(clf, X=M, y=L, cv=5, scoring=metric, n_jobs=-1)
        ret.update({'clf': clf,
                'clf_params': clf_hyper,
                metric: scores})
    return ret

#results = run(RandomForestClassifier, data, clf_hyper={})
#results2 = run2(RandomForestClassifier, data, clf_hyper={})



## Hypothetical Input, I will change later to use MCP
## Use classifier as key, not string.
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

## Next new hypothetical inputs
metrics = ['accuracy', 'roc_auc', 'recall', 'precision']


## Begin Cleaner Solution ...

class GridSearch(object):

    def __init__(self, X, y, models_params={}, metrics=[]):
        self.M = X
        self.L = y
        self.models_params = models_params
        self.metrics = metrics
        self.grid_results = []
        self.best_scores = []  # data structure used to print out scores
        self.best_metrics = {} # data structure used to plot/visualize

        # # Create an empty list of the same length as metrics, will use later in zip
        # # Is there a better way to initialize a dictionary?
        # ... There is ..
        # l = [0]*len(metrics)
        # z = [{}]*len(metrics)

        # Init with passed in clfs, and empty values for best scores, best params is a dictionary
        # for x in models_params:
        #     self.best_scores.append({'clf': x, 
        #                              'best_scores': dict(zip(metrics,l)),
        #                              'best_params': dict(zip(metrics,z))
        #                             })

        for x in models_params:
            self.best_scores.append({'clf': x, 
                                     'best_scores': dict.fromkeys(metrics,0),
                                     #'best_params': dict.fromkeys(metrics,{}),
                                     'best_params': {k: {} for k in metrics}
                                    })

        # Create container based on metric
        # This will be a list of best scores, in the order of input classifiers
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
            paramsToPassToRun = [dict(zip(keys, value)) for value in product(*values)]
            
            #Can probably use a list comprehension here
            for runParams in paramsToPassToRun:
                results = self.__run(model, data, runParams, metrics)
                self.grid_results.append(results)
        
        # And this too, a comprehension it could be
        #Update our metrics structure
        for model in self.best_scores:
            for metric in model['best_scores']:
                self.best_metrics[metric].append(model['best_scores'][metric])

        return self.best_scores

    # Our run implementation is a bit differen than what was assigned.
    # I changed it to use cross_val_score
    def __run(self, a_clf, data, clf_hyper={}, clf_metrics={}):
        clf = a_clf(**clf_hyper) # unpack parameters into clf as they exist

        ret = {}

        for metric in clf_metrics:
            scores = cross_val_score(clf, X=self.M, y=self.L, cv=5, scoring=metric, n_jobs=-1)
            ret.update({'clf': clf,
                        'clf_params': clf_hyper,
                        metric: scores})

            # Update our collection of best mean scores
            mean_score = scores.mean()
            for x in self.best_scores:
                if x['clf'] == a_clf and x['best_scores'][metric] < mean_score:
                    x['best_scores'][metric] = mean_score
                    x['best_params'][metric] = clf_hyper
        
        return ret


## Test Class
gs = GridSearch(M_std, L, model_params, metrics=metrics)
best_scores = gs.grid_search()

# Quick text dump of scores and params
for x in best_scores:
    print("")
    print(x['clf'].__name__)
    for metric in x['best_scores']:
        print(metric, "{:.2f}".format(x['best_scores'][metric]), "-", x['best_params'][metric])


print(gs.best_metrics)

# For Visualization, I imagine a panel for each metric, and a bar for each classifier showing the score.
# It would be really cool then if the tooltip over the bar for a classifier would show the best params.
# I'll leave out best params in the graph for now ...
#
# But I'd like to create a different dictionary structure for plotting.
# 
# Create classifier array of names
clfs_list = [x.__name__ for x in model_params]

import matplotlib.pyplot as plt

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
    
