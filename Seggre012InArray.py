"""
Segregate 0’s and 1’s in an array consisting of only 0’s and 1’s. ( O(n) approach)
"""

arr = [2, 0, 2, 1, 1, 0]  # [2, 2, 1, 0, 2, 0, 1]
print("the original list is :", arr)


def sort(arr):
    low = 0
    mid = 0
    high = len(arr) - 1

    while(mid < high):
        if(arr[mid] == 0):
            arr[low], arr[mid] = arr[mid], arr[low]
            mid += 1
            low += 1
        if(arr[mid] == 1):
            mid += 1
        if(arr[mid] == 2):
            arr[high], arr[mid] = arr[mid], arr[high]
            # mid += 1
            high -= 1

    print("the sorted order is :", end=" ")
    print(arr)


sort(arr)
