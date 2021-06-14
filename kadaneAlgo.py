"""
Largest Sum Contiguous Subarray
Difficulty Level : Medium
Last Updated : 13 May, 2021
Write an efficient program to find the sum of contiguous subarray within a one-dimensional array of numbers that has the largest sum. 

kadane-algorithm

 

Recommended: Please solve it on “PRACTICE ” first, before moving on to the solution. 
 
Kadane’s Algorithm:

Initialize:
    max_so_far = INT_MIN
    max_ending_here = 0

Loop for each element of the array
  (a) max_ending_here = max_ending_here + a[i]
  (b) if(max_so_far < max_ending_here)
            max_so_far = max_ending_here
  (c) if(max_ending_here < 0)
            max_ending_here = 0
return max_so_far
"""
import numpy as np
arr = [-2, -3, 4, -1, -2, 1, 5, -3]
meh = 0
msf = -np.inf
for i in arr:
    meh = meh + i
    if(meh < i):  # max ending
        meh = i
    if(msf < meh):  # max so far
        msf = meh

print(msf)
