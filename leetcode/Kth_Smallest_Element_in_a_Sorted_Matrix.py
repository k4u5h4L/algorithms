'''
Kth Smallest Element in a Sorted Matrix
Medium

Given an n x n matrix where each of the rows and columns are sorted in ascending order, return the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

 

Example 1:

Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13
Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13

Example 2:

Input: matrix = [[-5]], k = 1
Output: -5
'''

class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        def merge(arr1, arr2):
            l1 = 0
            l2 = 0
            
            res = []
            
            while l1 < len(arr1) and l2 < len(arr2):
                if arr1[l1] <= arr2[l2]:
                    res.append(arr1[l1])
                    l1 += 1
                else:
                    res.append(arr2[l2])
                    l2 += 1
                    
            while l1 < len(arr1):
                res.append(arr1[l1])
                l1 += 1
                
            while l2 < len(arr2):
                res.append(arr2[l2])
                l2 += 1
                    
            return res
        
        elements = []
        
        for arr in matrix:
            elements = merge(elements, arr)
            
        return elements[k-1]
