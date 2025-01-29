#fizzbuzz
# 1. Print numbers from 1 to 100
# 2. For multiples of 3, print "Fizz" instead of the number
# 3. For multiples of 5, print "Buzz" instead of the number
# 4. For multiples of 3 and 5, print "FizzBuzz" instead of the number
# 5. For all other numbers, print the number
while True:
    try:
        num = int(input("Enter a number: "))
        if num % 3 == 0 and num % 5 == 0:
            print("FizzBuzz")
        elif num % 3 == 0:
            print("Fizz")
        elif num % 5 == 0:
            print("Buzz")
        else:
            print(num)
    except ValueError:
        print("Please enter a valid number")
        continue
    else:
        break
def lengthOfLongestSubstring(s):
    start = maxLength = 0
    usedChar = {}
    for i in range(len(s)):
        if s[i] in usedChar and start <= usedChar[s[i]]:
            start = usedChar[s[i]] + 1
        else:
            maxLength = max(maxLength, i - start + 1)
        usedChar[s[i]] = i
    return maxLength

#leetcode -2 
#Given an integer x, return true if x is palindrome integer.
#An integer is a palindrome when it reads the same backward as forward. For example, 121 is palindrome while 123 is not.
#Example 1:
#Input: x = 121
#Output: true
#Example 2:
#Input: x = -121
#Output: false
#Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
def isPalindrome(x):
    if x < 0:
        return False
    else:
        return str(x) == str(x)[::-1]

def reverse(x):
    if x < 0:
        return -1 * int(str(x)[:0:-1]) if -1 * int(str(x)[:0:-1]) > -2**31 else 0
    else:
        return int(str(x)[::-1]) if int(str(x)[::-1]) < 2**31 - 1 else 0

with open('file.txt', 'r') as file:
    data = file.read().replace('\n', '')
    print(data)

#Machnine learning basics
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
#Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#learning Virtaul env
#conda create -n myenv python=3.6
#conda activate myenv
#conda deactivate
#conda remove -n myenv --all
#conda list