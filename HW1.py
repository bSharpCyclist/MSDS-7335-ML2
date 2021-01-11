#***********************
# Title: HW1
# Purpose: Understand Basic Python Data structures.
# 
# I enjoyed this exercise. I've been programming for many years now, in other languages.
# Python is refreshing to learn!
#
# Please note, this isn't polished. Things might not be consistent.
# Whatever code state my mind was in at the time, that's what I wrote.
# 
# Oh, and I may have not explored EVERY function - I tried to look at interesting ones though
# Author: Dan Crouthamel
# Date: January 2021
#***********************
import numpy as np
import datetime
import pprint as pprint

# Please fill in an explanation of each function and an example of how to use it below.

### Lists ###

## Append
print("")
print("Create an empty list and append a string object.")
theList = []
theList.append('String')
print(theList)

# Let's add an Integer to it, we can have different types in a list, that's nice.
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

# Python has error handling, that's good
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

## List Comprehensions -> They make sense when you think about what looping is happening.
# Link below is pretty good if you want a visual, which helps
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
## Nested Comprehensions - visualize what we are looping over, be the loop Danny!
print("\nNested Comprehensions - visualize what we are looping over, be the loop Danny!")
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

# Available methods - use help() or dir()
# ['clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']


# Think of key/value pairs
a_dict = {'I hate':'you', 'You should':'leave'}

# Values will return the values
print("\nWhat are the values in our dictionary?",a_dict.values())

# Keys will return the keys
print("What are the keys in our dictionary?",a_dict.keys())

# Items method returns a view object. The view object contains the key-value pairs of the dictionary, as tuples in a list.
print("What are the items in our dictionary?",a_dict.items())

# Has_Key is depracated. Use 'in' operator - it GRACEFULLY checks if a key exists. No bombs.
print("Is 'I hate' a key in our dictionary?",'I hate' in a_dict)

# Delete the Hate!
# Note, we could alternatively use Pop here.
# Pop-Item - Removes and returns (key, value) pair as 2-tuple.
del a_dict['I hate']
print("Our dictionary after deleting 'I hate' ->",a_dict)

# Update - We could use this to merge dictionaries and update existing keys
print("Adding {'I like':'you', and updating 'You should':'stay'} in our dictionary")
a_dict.update({'I like':'you', 'You should':'stay'})
print(a_dict)

# Clear - Removes all the elements from the dictionary
a_dict.clear()
print("Our dictionary after clearing it -> ",a_dict)

### Sets ###
# Sets are like dictionaries, but with keys only, no values. And since they are keys, they shoud be unqiue I think
# Sets are also unordered
#['add', 'clear', 'copy', 'difference', 'difference_update', 'discard', 'intersection', 'intersection_update']
#['isdisjoint', 'issubset', 'issuperset', 'pop', 'remove', 'symmetric_difference', 'symmetric_difference_update', 'union', 'update']
#
# We can create sets with set command or a set literal with curly braces
# Support mathematical operations like union, intersection, difference ... see below
#
# Create dummy sets
setA = {1,2,3,4,5,6,7,8,9,10}
setB = {6,7,8,9,10,11,12,13,15,15}
print("\nSets Section")
print("Set A", setA)
print("Set B", setB)

# Union of A & B should be all elements
# Use .union or |
print("Union of A & B", setA | setB)

# Intersection, as you would expect ..
print("Intersection of A & B" , setA.intersection(setB))

# What is in Set A, and not B?
# Note, you could use differenc_update to add those elements to A
print("What is in Set A, and not B?", setA.difference(setB))

# And the stuff that doesn't intersect?
# Symmmetric Difference ^
print("And the elements that don't intersect", setA ^ setB)

# Neither A or B is a sub or super set, these should all be false
print(setA.issubset(setB),setA.issuperset(setB))

### Strings ###
print("\nString Section")
# I'm not going to go through these if that's OK, I've used very similar functions elsewhere in time.
# I make use of some string properties later in the code section

#  'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 
#  'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 
#  'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 
#  'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']

### Collections ###
print("\nCollections Section")
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
plt.show()


# Itertools - Functions for creating iterators for efficient looping. Very nice!
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
# This should be the same as above, but with 3 colors instead of 2 .. right?

## Create the combinations, use counter to order and print out
print("")
colorCombinations = [y for x in flower_orders for y in combinations(x.split('/'),3)]
colorCounter = Counter(colorCombinations)
for i in colorCounter:
    print("% s : % s" % (i, colorCounter[i]), end ="\n")


# 6. Make dictionary of where keys are a color and values are what colors go with it
# Didn't really follow what was being asked, so I'll make something up. We will answer this question.
# When a color X appears in a on order, it appeared with the following other colors.

# We can use set compreshenions too, intialize to empty set for each color
print("")
uniqueColors = set(colors)
newDict = {i : set() for i in uniqueColors}

# Come back to this, I think we can write it better ...
for order in flower_orders:
    orderList = order.split('/')
    for color in orderList:
        orderSet = set(orderList)
        orderSet.remove(color)
        # union we existing value
        newDict[color] = orderSet | newDict[color]

pprint.pprint("Pretty Print Dictionary")
pprint.pprint(newDict)


import string
dead_men_tell_taies = ['Four score and seven years ago our fathers brought forth on this',
'continent a new nation, conceived in liberty and dedicated to the',
'proposition that all men are created equal. Now we are engaged in',
'a great civil war, testing whether that nation or any nation so',
'conceived and so dedicated can long endure. We are met on a great',
'battlefield of that war. We have come to dedicate a portion of',
'that field as a final resting-place for those who here gave their',
'lives that that nation might live. It is altogether fitting and',
'proper that we should do this. But in a larger sense, we cannot',
'dedicate, we cannot consecrate, we cannot hallow this ground.',
'The brave men, living and dead who struggled here have consecrated',
'it far above our poor power to add or detract. The world will',
'little note nor long remember what we say here, but it can never',
'forget what they did here. It is for us the living rather to be',
'dedicated here to the unfinished work which they who fought here',
'have thus far so nobly advanced. It is rather for us to be here',
'dedicated to the great task remaining before us--that from these',
'honored dead we take increased devotion to that cause for which',
'they gave the last full measure of devotion--that we here highly',
'resolve that these dead shall not have died in vain, that this',
'nation under God shall have a new birth of freedom, and that',
'government of the people, by the people, for the people shall',
'not perish from the earth.']

# 1. Join everything

# Loop over and join, using a generator expression
# https://note.nkmk.me/en/python-string-concat
y = ''.join(str(n) for n in dead_men_tell_taies)
print('')
print(y)

# 2. Remove spaces
# Just going to use a replace function
nospaces = y.replace(' ', '')
print('')
print(nospaces)

# 3. Occurrence probabilities for letters
# Our string includes upper and lower case letters and punctuation.
# I will convert to lower case, and remove punctuation
cleanStr = nospaces.lower()
cleanStr = "".join(str(n) for n in cleanStr if n in string.ascii_lowercase)

# Let's create a counter, and then normalize by the length of the new string, *100
c = Counter(cleanStr)
numletters = len(cleanStr)

print('')
for key in sorted(c, key=c.get, reverse=True):
    print(key,"appears",c[key]/numletters*100,"% of the time.")


# 4. Tell me transition probabilities for every letter pairs
#
# Had to research this a bit. Sounds like we need to create a 26x26 matrix
# 
tm = np.zeros((26,26))

# I could probably create an interable that returns every 2 characters? Not sure
for index,letter in enumerate(cleanStr):
    # Ignore last letter
    if index==(len(cleanStr)-1):
        exit
    else:
        x = string.ascii_lowercase.index(letter)
        y = string.ascii_lowercase.index(cleanStr[index+1])
        tm[x,y] = tm[x,y] + 1

# To calculate the probabilities, we would divide a cell value by the row total.
# Note, division by zero will result in nan, so replace them.
tm = tm / tm.sum(axis=1, keepdims=True)
tm = np.nan_to_num(tm,)

print("")
print(tm)

# optional
# 5. Plot graph of transition probabilites from letter to letter
# I'm ploting the probabilities for B only A = 0, Z = 25
# I could throw this into a loop, show all letters, etc. Come back later if time ...
# The most commone letter after B was E, which is to be expected.
plt.plot(tm[1,:])
plt.title("Probability of a letter appearing after 'b'")
plt.ylabel("Probabilty")
plt.xticks(np.arange(26), string.ascii_lowercase)
plt.xlabel("Letter")
plt.show()

# Do we get fancy, and try a heat map? Of course.
plt.title("Letter Transition Probabilities")
plt.imshow(tm, cmap='hot', interpolation='nearest')
plt.xticks(np.arange(26), string.ascii_lowercase)
plt.xlabel("Letter Transitioning To")
plt.yticks(np.arange(26), string.ascii_lowercase)
plt.ylabel("Letter Transitioning From")
plt.show()


# 7. Flatten a nested list
# How many nested levels? Sounds like a recurrsion problem to me.
# There are several ways you could do this. List comprehension would work too
# for a limited number of levels.
# There is itterools, etc
# The below seemed interesting
# https://stackoverflow.com/questions/12472338/flattening-a-list-recursively


def flatten(list_of_lists):
    # End recurssion
    if list_of_lists == []:
        return list_of_lists
    # Is Element a list? Recursively flatten it and the rest
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    # Element is not a list, just return it (as a list) and flatten the remainder
    return list_of_lists[:1] + flatten(list_of_lists[1:])

regular_list = ['String',[1,"A","Roger"],[2,["A","B","C",["Another Nested Level","Crazy"]],"End 2"],[3,4],"What","Stop!"]

print('Original list', regular_list)
print('Transformed list', flatten(regular_list))
