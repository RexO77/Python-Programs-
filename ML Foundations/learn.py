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