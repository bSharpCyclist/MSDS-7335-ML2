#***********************
# Title: HW1
# Purpose: Homework 1 Assignment
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

## List Comprehensions
