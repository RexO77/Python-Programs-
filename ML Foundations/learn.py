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
    