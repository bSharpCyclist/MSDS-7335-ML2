#***********************
# Title: HW1
# Purpose: Understand Basic Python Data structures. That's it.
# Author: Dan Crouthamel
# Date: January 2021
#***********************
import numpy as np
import datetime

# Please fill in an explanation of each function and an example of how to use it below.

### Lists ###

## Append
print("")
print("Create an empty list and append a string object.")
theList = []
theList.append('String')
print(theList)

# Let's add an Integer to it
print("\nNow let's add an integer to the list.")
theList = theList + [1]
print(theList)

## Extend
print("\nNow let's extend our list by passing another list of more than 1 element.")
extendList = ['a', 'b', 7, 11.2]
theList.extend(extendList)
print(theList)

## Index - Returns the index of a specified element in a list
print("\nUsing the index function, we can find the position of a given item.")
print("Let's find where 7 is.")
print("The index of 7 is:", theList.index(7))

print('\nWe can also specify a starting postion for the index function')
print("If we start at the 5th element will we find 7? Probably not.")
print("Let's find where 7 is starting from 5.")

try:
    position = theList.index(7,5)
    print("The index of 7 starting from 5 is:", position)
except ValueError:
    print("We could not find 7 starting from 5.")
print("As expected, a Value Error occured")

## Insert
print("\nLet's insert a datetime object into the list.")
dateTimeObj = datetime.datetime.now()
theList.insert(3,dateTimeObj)
print(theList)

## Remove
print("\nNow let's remove that object.")
theList.remove(dateTimeObj)
print(theList)

## POP
print("\nPop will return an item for a given index, and also remove the object from the list.")
print("Let's pop off 'b'")
pop = theList.pop(theList.index('b'))
print(pop)
print(theList)

print("\nUsing pop with no argument will pop off the last element of the list.")
last = theList.pop()
print(last)
print(theList)

## Concatenation, Multiplication
print("\nUsing '+' will concatenate one list with another.")
print("[1] + [1] =",[1]+[1])

print("\nUsing '*' will concatenate to the list n times")
print("[1,2] * 3 =", [1,2]*3)

## Slicing
print("\nWe can slice our lists too, like so <list>[a:b]")
print("<list>[1:] would return everything but the first element.")
print("[1,2][1:] =", [1,2][1:])
print("theList[1:] = ",theList[1:])

## List Comprehensions -> They make sense, to me, when you think about what looping is happening.
# Link below is pretty good
# https://towardsdatascience.com/11-examples-to-master-python-list-comprehensions-33c681b56212
# [expr for val in collection if condition]
#
# result = []
# for val in collection:
#     if condition:
#         result.append(expr)
#
# [x for x in [2,3]] -> we are just duplicating the list here
print("\n[x for x in [2,3]] -> This just duplicates the list")
print("[x for x in [2,3]] =",[x for x in [2,3]])
# 
# [x for x in [1,2] if x ==1] -> return me a new list of only those elements that equal 1.
print("\n[x for x in [1,2] if x ==1] -> Return me a new list of only those elements that equal 1.")
print("[x for x in [1,2] if x ==1] =",[x for x in [1,2] if x ==1])
#
#
## Nested Comprehensions
# [y*2 for x in [[1,2],[3,4]] for y in x]
# This creates a new list where each element is squared.
# 
# for in x in [[1,2],[3,4]]
#     for y in x
#        list.append(y^2) 
print("\n[y*2 for x in [[1,2],[3,4]] for y in x] -> Create a new list containing each entry in the matrix squared.")
print("[y*2 for x in [[1,2],[3,4]] for y in x] = ",[y*2 for x in [[1,2],[3,4]] for y in x])
#
#
### Tuple ###
theTuple = (1,2,3,7,1,2,3,4,1,2,1,2,1,2,3,1,4,4,1)
## Count method will return the frequency of an element
print("\nThe value 4 appears",theTuple.count(4),"times.")
## Index method will return the element at a given position, the first occurence
print("The first occurence of 4 is at index", theTuple.index(4),"Remeber, indexing starts with 0.")


### Dictionary ###
# Think of key/value pairs
a_dict = {'I hate':'you', 'You should':'leave'}

# Values will return the values
print("\nWhat are the values in our dictionary?",a_dict.values())

# Keys will return the keys
print("What are the keys in our dictionary?",a_dict.keys())

# Items method returns a view object. The view object contains the key-value pairs of the dictionary, as tuples in a list.
print("What are the items in our dictionary?",a_dict.items())

# Has_Key is depracated. Use something like ‘never’ in a_dict
print("Is 'I hate' a key in our dictionary?",'I hate' in a_dict)

# Delete a key
del a_dict['I hate']
print("Our dictionary after deleting 'I hate' ->",a_dict)

# Clear - Removes all the elements from the dictionary
a_dict.clear()
print("Our dictionary after clearing it -> ",a_dict)

### Sets ###
# Sets are like dictionaries, but with keys only, no values.
#['add', 'clear', 'copy', 'difference', 'difference_update', 'discard', 'intersection', 'intersection_update']
#['isdisjoint', 'issubset', 'issuperset', 'pop', 'remove', 'symmetric_difference', 'symmetric_difference_update', 'union', 'update']
theSet = {'Fireb'}



### Strings ###


### Collections ###
from collections import Counter
#['clear', 'copy', 'elements', 'fromkeys', 'get', 'items', 'keys', 'most_common', 'pop', 'popitem', 'setdefault', 'subtract', 'update', 'values']

# Create an empty counter object
c = Counter()

# We can use update to add some items to it.
# Take a look at the output .. what happened?
# IMPORTANT -> Elements are stored as dictionary keys and their counts are stored as dictionary values.
c.update([1,2,7,9,5,5,6,2,3,1,8,8,9,4,3,6,7,1,5,4,2,3,1])
print("\nOur Counter Object ->",c)

# What are the two most common elements?
print("The 2 most common elements in our counter are ->",c.most_common(2))
print("1 appears 4x and 2 appears 3x")

# Values will return all values. Let's sum them up.
print("The sum of all our values in our Counter ->",sum(c.values()))

# Clear will empty the Counter
c.clear()

flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
'W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R',
'W/R','W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V',
'W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V',
'W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','B/Y','B/Y',
'B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y',
'W/N/R/B/V/Y','W/G','W/G','W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V',
'W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y','W/V/O',
'W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y','W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y',
'W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y',
'W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R',
'V/Y','V','N/R/V','N/V/Y','R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y',
'N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O','N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P',
'W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']

# 1. Build a counter object and use the counter and confirm they have the same values.
print("\n1. Build a counter object and use the counter and confirm they have the same values.")
# The counter object gives a unique set of all flowers orders and frequency
flwCnt = Counter(flower_orders)

# If we use the set operator on flower orders we will also get a unique set, but no frequency
# So let's compare the unique keys in the counter with those in a set. Sort both first though
flwSet = set(flower_orders)

# Convert both to lists, sort, and compare
print("Counter has the same orders?",list(flwCnt.keys()).sort() == list(flwSet).sort())

# 2. Count how many objects have color W in them.
# I wanted to write a foor loop, but I think we can use a list comprehension instead
print("\n2. Count how many objects have color W in them.")

Worders = [x for x in flwCnt if 'W' in x]
print("The number of orders having 'W' in them:",len(Worders))


# 3. Make histogram of colors
# I'm assuming individual colors
# Nested comprehensions!

# This would print out all colors
#for x in flower_orders:
#    for y in x.split('/'):
#        print(y)

# Nested List Comprehension
from matplotlib import pyplot as plt
colors = [y for x in flower_orders for y in x.split('/')]
plt.hist(colors)
#plt.show()


# Itertools - Functions for creating iterators for efficient looping
# https://docs.python.org/3/library/itertools.html

# 4. Rank the tuples of color pairs regardless of how many colors in order.
from itertools import *

## Pseduo code ... , tranlate to LC
""" for x in flower_orders:
    print(x)
    for y in combinations(x.split('/'),2):
        print(y) """

## Create the combinations, use counter to order and print out
print("")
colorCombinations = [y for x in flower_orders for y in combinations(x.split('/'),2)]
colorCounter = Counter(colorCombinations)
for i in colorCounter:
    print("% s : % s" % (i, colorCounter[i]), end ="\n")



# 5. Rank the triples of color pairs regardless of how many colors in order.
# This should be the same as above, but with 3 colors instead of 2

## Create the combinations, use counter to order and print out
print("")
colorCombinations = [y for x in flower_orders for y in combinations(x.split('/'),3)]
colorCounter = Counter(colorCombinations)
for i in colorCounter:
    print("% s : % s" % (i, colorCounter[i]), end ="\n")