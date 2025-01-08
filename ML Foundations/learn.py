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