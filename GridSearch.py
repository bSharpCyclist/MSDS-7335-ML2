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

results = run(RandomForestClassifier, data, clf_hyper={})
results2 = run2(RandomForestClassifier, data, clf_hyper={})

# Let's test the 'run' method with a different classifier and parameters
mlp_params = {
    'solver': 'lbfgs',
    'alpha': 1e-5,
    'hidden_layer_sizes': (2, 3),
    'random_state': 1,
    'max_iter': 100
}
results = run(MLPClassifier, data, clf_hyper=mlp_params)

print(results)

## Ideally we need to put all of this into a class with callable methods.

## Hypothetical Input, I will change later to use MCP
## Use classifier as key, not string.
model_params = {
    RandomForestClassifier: { 
        #"n_estimators" : [100, 200, 500, 1000],
        "n_estimators" : [100, 200],
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

# This will hold all of our results
# It will be a list of dictionaries. A dictionary has the classifier, params, and the accuracy score, for now
# We need to compare other metrics too
gridResults = []

## Hi Fabio, this is a bit verbose, but I'm tyring to explain what is happening.
## It helps to step through code and take a look at the variables, their types, vals, etc.
## You can do this in VSCode, and keep variables list up. Take a lok at the different data structures
for model, params in model_params.items():
    break

    # This will hold all of our results
    # It will be a list of dictionaries. A dictionary has the classifier, params, and the accuracy score, for now
    # We need to compare other metrics too
    #gridResults = []

    print("")
    print("What model?",model)

    # We can do this to see each paramter and the associated values
    for param_key, param_values in params.items():
        print("")
        print("Parameter Key and Values",param_key, param_values)

    # Tuples ('solver', 'multi_class') (['newton-cg', 'sag', 'lbfgs'], ['ovr', 'multinomial'])
    # What are we doing here?
    # Create tuples for all the keys (key1, key2, key3) and the values for those keys
    # Keys = tuples of strings (ParamName)
    # Values = tuples of lists of parameter values
    # It's the values we'll want to use in the product function
    keys, values = zip(*params.items())
    print("")
    print("Tuples",keys,values)

    # We can upack our values as a parameter to the product function to
    # find all the different combinations.
    for x in product(*values):
        print("")
        print("Using Product Function", x)

    # Look at the output. We have all the different permutations.
    # Our run method expects a dictionary of param names and values
    # So let's loop over all the permutations, zip that with our keys to create tuples of param/paramValue
    # And then create a dictionary out of those tuples. Does that make sense?
    # ParamsToPassToRun is a list of dictionaries, and the run method wants a dictionary. Will then iterate over that.
    paramsToPassToRun = [dict(zip(keys, value)) for value in product(*values)]
    print("")
    print("Params To Pass To Run", paramsToPassToRun)

    ## Now loop over the paramsToPass
    # We could use list comprehensions here, but I'm being verbose to start
    # so that we can understand what is happening.
    for runParams in paramsToPassToRun:
      print(runParams)

      ## Old Run Method
      #results = run(model, data, runParams)
      #gridResults.append(results)

      ## Use modified run, to support mutliple metrics
      results = run2(model, data, runParams, metrics)
      gridResults.append(results)

      ## Will then later process gridResults to get best scores, visualize, plot, etc.

print("")
print(len(gridResults))


## Begin Cleaner Solution ...

class GridSearch(object):

    def __init__(self, X, y, models_params={}, metrics=[]):
        self.M = X
        self.L = y
        self.models_params = models_params
        self.metrics = metrics
        self.gridResults = []

    def grid_search(self):
        for model, params in self.models_params.items():
            print("")
            print("Evaluating ...", model.__name__, "...")
            #
            keys, values = zip(*params.items())
            paramsToPassToRun = [dict(zip(keys, value)) for value in product(*values)]
            
            #
            for runParams in paramsToPassToRun:
                results = self.__run(model, data, runParams, metrics)
                gridResults.append(results)
            
        return gridResults

    def __run(self, a_clf, data, clf_hyper={}, clf_metrics={}):
        clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist
        #M, L = data

        ret = {}

        for metric in clf_metrics:
            scores = cross_val_score(clf, X=self.M, y=self.L, cv=5, scoring=metric, n_jobs=-1)
            ret.update({'clf': clf,
                    'clf_params': clf_hyper,
                    metric: scores})
        return ret

## Test Class
gs = GridSearch(M_std, L, model_params, metrics=metrics)
results = gs.grid_search()
print(len(results))
