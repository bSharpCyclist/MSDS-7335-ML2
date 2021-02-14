#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:47:40 2021

@author: fabiosavorgnan
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


people = {'Jane': {'willingness to travel': 1,
                  'desire for new experience':5,
                  'cost':3,
                  'average rating':1,
                  'cusine':3,
                  'vegetarian': 2,
                  },
          'Bob': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':3,
                  'average rating':1,
                  'cusine':3,
                  'vegetarian': 2,
                  },
         'Paul': {'willingness to travel': 2,
                  'desire for new experience':4,
                  'cost':2,
                  'average rating':2,
                  'cusine':4,
                  'vegetarian': 3,
                  },
          'Jim': {'willingness to travel': 3,
                  'desire for new experience':2,
                  'cost':2,
                  'average rating':3,
                  'cusine':1,
                  'vegetarian': 2,
                  },
                  
        'Kasi': {'willingness to travel': 3,
                  'desire for new experience':4,
                  'cost':2,
                  'average rating':1,
                  'cusine':3,
                  'vegetarian': 3,
                  },
          'Ron': {'willingness to travel': 4,
                  'desire for new experience':3,
                  'cost':2,
                  'average rating':3,
                  'cusine':2,
                  'vegetarian': 3,
                  },
         'Joe': {'willingness to travel': 2,
                  'desire for new experience':4,
                  'cost':2,
                  'average rating':2,
                  'cusine':3,
                  'vegetarian': 3,
                  },
          'Alex': {'willingness to travel': 2,
                  'desire for new experience':4,
                  'cost':2,
                  'average rating':4,
                  'cusine':5,
                  'vegetarian': 1,
                  },
         'Fabio':{'willingness to travel': 4,
                  'desire for new experience':3,
                  'cost':2,
                  'average rating':2,
                  'cusine':2,
                  'vegetarian':1,
                  },
          'Saul': {'willingness to travel': 1,
                  'desire for new experience':4,
                  'cost':2,
                  'average rating':2,
                  'cusine':3,
                  'vegetarian': 3,
                  }
}
          
restaurants  = {'flacos':{'distance' : 2,
                          'novelty' : 3,
                          'cost': 4,
                          'average rating': 5,
                          'cuisine': 5,
                          'vegetarian': 5
                          },
                'Joes':{'distance': 5,
                        'novelty' : 1,
                        'cost': 5,
                        'average rating': 5,
                        'cuisine': 5,
                        'vegetarian': 3
                          },
                  'AB':{'distance' : 3,
                          'novelty' : 2,
                          'cost': 4,
                          'average rating': 3,
                          'cuisine': 2,
                          'vegetarian': 1
                          },
                 'AC':{'distance' : 1,
                        'novelty' : 2,
                        'cost': 4,
                        'average rating': 3,
                        'cuisine': 2,
                        'vegetarian': 1
                          },
                 'Sushi':{'distance' : 1,
                          'novelty' : 3,
                          'cost': 2,
                          'average rating': 4,
                          'cuisine': 3,
                          'vegetarian': 2
                          },
                'Italian':{'distance' : 2,
                          'novelty' : 5,
                          'cost': 3,
                          'average rating': 2,
                          'cuisine': 1,
                          'vegetarian': 1
                          },
                'Arab':{'distance': 3,
                        'novelty' : 2,
                        'cost': 4,
                        'average rating': 3,
                        'cuisine': 2,
                        'vegetarian': 1
                          },
                  'Az':{'distance' : 1,
                          'novelty' : 1,
                          'cost': 1,
                          'average rating': 1,
                          'cuisine': 2,
                          'vegetarian': 1
                          },
                 'At':{'distance' : 3,
                        'novelty' : 5,
                        'cost': 5,
                        'average rating': 1,
                        'cuisine': 2,
                        'vegetarian': 1
                          },
                'Chinese':{'distance' : 2,
                          'novelty' : 3,
                          'cost': 5,
                          'average rating': 3,
                          'cuisine': 2,
                          'vegetarian': 1
                          }
                       
} 
 # People matrix               
M_people = np.array([[v2 for _, v2 in v.items()] for _, v in people.items()])                
M_people 


# Restaurant matrix
M_restaurants= np.array([[v2 for _, v2 in v.items()] for _, v in restaurants.items()])
M_restaurants           

# Definition of linear combination
# If you take a set of matrices, you multiply each of them by a scalar, 
# and you add together all the products thus obtained, then you obtain a linear combination.
# Note that all the matrices involved in a linear combination need to have the same dimension 
# (otherwise matrix addition would not be possible)

# We will multiply the matrix choose from the peoples or user to the information that we got from the
# website about the restaurant in order to find the recommended restaurant

m= M_people[0]
m 
np.dot(M_restaurants,m)
# Each entry in the producing vector represent the sume of the score for each restaurant 
# for the specific user

from scipy.stats import rankdata

rankdata(np.dot(M_restaurants,m))
# Higher rank better
# The restaurant number 1 (flacos) has the highest score and it is the preferred restaurant for the selected user 
# The number 9 restaurant is the second in the rank to be choose by the selected user,(restaurant= At) 

# Better display of the rank
rankdata(len(m)-rankdata(np.dot(M_restaurants,m)))


# Multiples user
def get_real_rank(data):
  return rankdata(len(data)-rankdata(data))

# This matrix the column are the restaurant and the row are the user
M_usr_x_rest= np.dot(M_restaurants,M_people.T)
M_usr_x_rest

# Ech entry in this vector is the score for the choosing restaurant from all the user. Higher is better
sum(np.dot(M_restaurants,M_people.T).T)

# Explore the output
np.dot(M_restaurants,M_people.T).T[:,0]

# Get the sum of the score for the first user
sum(np.dot(M_restaurants,M_people.T).T[:,0])

# We will get the rank of the restaurant. Lower rank better
# Restaurant flacos is the number one ranked and restaurant, and restaurant Joe is the second. 
get_real_rank(sum(np.dot(M_restaurants,M_people.T).T))
#`````````````````````````````````````````````````````````````````````````````````````````````````````

# We will calculate the user rank by the row 
M_usr_x_rest_rank= []
for row in M_usr_x_rest:
    ranked_row = get_real_rank(row)
    M_usr_x_rest_rank= ranked_row
    M_usr_x_rest_rank = np.array(M_usr_x_rest_rank)
    

print( M_usr_x_rest_rank)
