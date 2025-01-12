my_dict = {'one': 1, 'two': 2, 'three': 3}
print(my_dict['one'])
print(my_dict['two'])
tuples = ('Hello, World', 2)

# Print the tuple
print("Tuple:", tuples)  

# Count occurrences of 2 and print the result
count_of_2 = tuples.count(2)
print("Count of 2:", count_of_2)  

# Find the index of 2 and print the result
index_of_2 = tuples.index(2)
print("Index of 2:", index_of_2)

# Correct file handling
with open('file.txt', 'w') as file:
    file.write('Hello, World!')
    file.write('This is the new line')  # Write to file

# Open file again for reading
with open('file.txt', 'r') as file:
    content = file.read()  # Read the content
    print("File content:", content)

d = {'k1':{'k2':'hello'}}
# Grab 'hello'
d['k1']['k2']

# two nested lists
l_one = [1,2,[3,4]]
l_two = [1,2,{'k1':4}]

# True or False?
l_one[2][0] >= l_two[2]['k1']
# Getting a little tricker
d = {'k1':[{'nest_key':['this is deep',['hello']]}]}

#Grab hello
d['k1'][0]['nest_key'][1][0]

dict = {'test': 1, 'test2': 2}
print(dict['test'])
my_dict['key3'][0].upper()
print(my_dict)
while True:
    try:
        x = int(input("Please enter a number: "))
        break
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")