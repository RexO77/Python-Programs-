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

#leetcode -3 
#Given a string s, find the length of the longest substring without repeating characters.
#Example 1:
#Input: s = "abcabcbb"
#Output: 3
#Explanation: The answer is "abc", with the length of 3.
#Example 2:
#Input: s = "bbbbb"
#Output: 1
#Explanation: The answer is "b", with the length of 1.
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

#leet code -3
#Given an integer x, return x with its digits reversed.
#If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
#Assume the environment does not allow you to store 64-bit integers (signed or unsigned).
#Example 1:
#Input: x = 123
#Output: 321
#Example 2:
#Input: x = -123
#Output: -321
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
