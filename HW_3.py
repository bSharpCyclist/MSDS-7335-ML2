#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:47:40 2021

@author: fabiosavorgnan
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import iqr


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
sum(np.dot(M_restaurants,M_people.T))

# Explore the output
np.dot(M_restaurants,M_people.T)[:,0]

# Get the sum of the score for the first user
sum(np.dot(M_restaurants,M_people.T)[:,0])

# We will get the rank of the restaurant. Lower rank better
# Restaurant Joes is the number one ranked and restaurant, and restaurant Az is the second. 
get_real_rank(sum(np.dot(M_restaurants,M_people.T)))
#`````````````````````````````````````````````````````````````````````````````````````````````````````
# Why is there a difference between the two? 
# array([411, 503, 443, 347, 410, 440, 417, 495, 376, 394])score
#array([ 6.,  1.,  3., 10.,  7.,  4.,  5.,  2.,  9.,  8.]) rank
# Yes, there is a diference between the 2

# What problem arrives?  What does it represent in the real world? 
# The problem that arrives is that the diference in the score between the second and the third is 
# 52, and the diference between the first and the second is only 8. This can be misleading in the real
# world if somebody is just answering question without though and that person can change the rank 
# in a direction that will not reflex the reality of the business.

# How should you preprocess your data to remove this problem. 
# I would set up an screening to find out the people that does not give a honest answer. Afterword, I would proceed
# to remove those people from the data in order to avoid the problem stated in the previous question.


# Find  user profiles that are problematic, explain why?
# In our data set there was not any particular user that was a problem, but very well it could be a problem
# in the real wrold.
people_name= people.keys()
people_name= list(people_name)
people_name

user_avg= M_usr_x_rest.mean(axis= 1)
user_std= M_usr_x_rest.std(axis= 1)
user_variance= user_std/user_avg 

for i, x in zip(people_name,user_variance ):
    print(i + ' = ' + str(x))


plt.subplots(figsize=(10,10))
plt.boxplot(x= user_variance, showmeans= True)
plt.ylabel("Variance")
plt.xlabel("Users")
plt.title("Boxplot of user variance")
plt.show()

# Think of two metrics to compute the disatistifaction with the group.
# We will look at the diference between the highest score from the best restaurant and all the other 
# score from the same restaurant and we will find the STD and iqt of that difference. This proces we will also do for the other
# restaurant.
# We will do specifially for the best restaurant and the worst restaurant.  
rest_name= restaurants.keys()
rest_name= list(rest_name)
rest_name
rest_rating= M_usr_x_rest.sum(axis=0)
rest_rating
rest_best= M_usr_x_rest[np.argmax(rest_rating),:]
rest_best
best_rest= rest_name[np.argmax(rest_rating)]
best_rest
rest_best_max= np.max(rest_best)
disatisfaction= rest_best_max - rest_best
disatisfaction
dis_std= np.std(disatisfaction)
dis_std
dis_iqt= iqr(disatisfaction)
dis_iqt
print("Restaurants name =",rest_name )
print("Individualized restaurnat rating =", rest_rating)
print("Best restaurant rating =",rest_best )
print("The best restaurant for the entire group =",best_rest)
print("Now we look at the disatisfaction of the whole group")
print("Standard desviation of the whole group =",dis_std )
print("IQR of the whole group =",dis_iqt)

rest_worst= M_usr_x_rest[np.argmin(rest_rating),:]
rest_worst
worst_rest= rest_name[np.argmin(rest_rating)]
worst_rest
rest_worst_m= np.min(rest_worst)
disatisfaction2= rest_worst - rest_worst_m
disatisfaction2
dis2_std= np.std(disatisfaction2)
dis2_std
dis2_iqt= iqr(disatisfaction2)
dis2_iqt

print("Restaurants name =",rest_name )
print("Individualized restaurnat rating =", rest_rating)
print("Worst restaurant rating =",rest_worst )
print("The worst restaurant for the entire group =", worst_rest)
print("Now we look at the disatisfaction of the whole group")
print("Standard desviation of the whole group =",dis2_std )
print("IQR of the whole group =",dis2_iqt)

# The best restaurant for the entire group = Joes
# Now we look at the disatisfaction of the whole group with respect to the best restaurant
# Standard desviation of the whole group = 7.249827584156743
# IQR of the whole group = 12.25
# The worst restaurant for the entire group = AC
# Now we look at the disatisfaction of the whole group with respect to the worst restautant
# Standard desviation of the whole group = 3.721558813185679
# IQR of the whole group = 3.5

# Here are the variability from the user for the others restaurants for completness
rest_name
disatisfaction= []
dis_std= []
dis_iqt=[]

for rows, i in zip(M_usr_x_rest, rest_name):
    print("Restaurant= ", i)
    row_max= np.max(rows)
    disatisfaction = row_max - rows
    print("Variability among the user =", disatisfaction)
    dis_std = np.std(disatisfaction)
    print("Standard Desviation=", dis_std)
    dis_iqt = iqr(disatisfaction)
    print("Interquartile range among the user =", dis_iqt)

# Ok. Now you just found out the boss is paying for the meal. 
# How should you adjust? Now what is the best restaurant?

# People matrix without the cost                              
M_boss_p= M_people.copy()
M_boss_p[:,2]= 0
M_boss_p

M_usr_B_x_rest= np.dot(M_restaurants,M_boss_p.T)
M_usr_B_x_rest

# We will get the rank of the restaurant. Lower rank better
# Restaurant Az is the number one ranked and restaurant, Joe is the second.
# Surprisingly the boss paying the bill make the first and the second choices flip in the choosing of the restaurant
get_real_rank(sum(np.dot(M_restaurants,M_boss_p.T)))

# Should you split in two groups today? 

# cluster to see the separation of the groups
plt.style.use("ggplot")
X1 = M_people 

plt.scatter(X1[:, 1], X1[:, 0]+np.random.random(X1[:, 1].shape)/2, 
             s=20)
plt.xlabel('Peoples'), plt.ylabel('Choices')
plt.grid()
plt.title('People choices')
plt.show()

from sklearn.cluster import KMeans
labellist= people_name
n_fare= 2
X1 = M_people 

cls_fare = KMeans(n_clusters=n_fare, init='k-means++',random_state=1)
cls_fare.fit(X1)
newfeature_fare = cls_fare.labels_ # the labels from kmeans clustering
centrioid= cls_fare.cluster_centers_

# We can see here the separation of the groups with the application of Knn
plt.figure()
plt.subplot(1,2,1)
X1=X1
plt.scatter(X1[:, 0], X1[:, 1]+np.random.random(X1[:, 1].shape)/2, c=newfeature_fare, cmap=plt.cm.rainbow, s=20, linewidths=0)
plt.xlabel('People'), plt.ylabel('Choices')
plt.grid()

# We then find the people in the groups

colors= ['g', 'r']
plt.figure()
plt.subplot(1,2,1)
for i in range(len(X1)):
    plt.plot(X1[i][0], X1[i][1], colors[newfeature_fare[i]],markersize= 15 ) 
    plt.annotate(labellist[i],(X1[i][0], X1[i][1]), size= 10)
plt.scatter(centrioid[:,0],centrioid[:,1],marker= 'o', s=150, linewidths= 2, zorder= 10)
plt.xlabel('People'), plt.ylabel('Choices')
plt.grid()

# I believe we can split in 2, one would be the Saul, kasi and the other around them. And the other is Fabio with the 
# other guys close to him, but it would not work for jane, Bob and jim, they are far apart.

# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  
# Can you find their weight matrix? 

# We will not be able to find the weight matrix because we dont have the raw datafrom the new team. We only have their rank 
# of the restaurant but we dont have the detail raw data that gave those rank of the restaurant. The weight matrix would be the 
# multiplication of the R_matrix against the new team matrix of the detailed preference of the new team of the restaurant.



# extra code
disatisfaction= np.zeros((10,10))
dis_std= np.zeros(10)
dis_iqt=np.zeros(10)
for counter,row in enumerate(M_usr_x_rest):
    row_max= np.max(row)
    disatisfaction[counter,:] = row_max - row
    dis_std[counter] = np.std(disatisfaction[counter,:] )
    dis_iqt[counter] = iqr(disatisfaction[counter,:] )
       
print(disatisfaction, dis_std, dis_iqt)