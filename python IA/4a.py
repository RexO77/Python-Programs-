import random
def mergeSort(list):
    if len(list)>1:
        mid = len(list)//2
        left = list[:mid]
        right = list[mid:]
        mergeSort(left)
        mergeSort(right)
        i=j=k=0
        while i<len(left) and j<len(right):
            if left[i]<right[j]:
                list[k] = left[i]
                i+=1
            else:
                list[k] = right[j]
                j+=1
            k+=1
        while i<len(left):
            list[k] = left[i]
            i+=1
            k+=1
        while j<len(right):
            list[k]=right[j]
            j+=1
            k+=1
        return list
def insertionSort(arr):
    for i in range(10):
        key = arr[i]
        j=i-1
        while j>=0 and key<arr[j]:
            arr[j+1]=arr[j]
            j-=1
            arr[j+1]=key
myList = []
for i in range(10):
    myList.append(random.randint(0,999))
print("Unsorted List ")
print(myList)
mergeSort(myList)
print("Sorting Using Merge Sort")
print(myList)
print("-----------------------------")
myList = []
for i in range(10):
    myList.append(random.randint(0,999))
print("Unsorted List ")
print(myList)
insertionSort(myList)
print("Sorting Using Insertion Sort")
print(myList)
