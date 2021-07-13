'''
Element Appearing More Than 25% In Sorted Array
Easy

Given an integer array sorted in non-decreasing order, there is exactly one integer in the array that occurs 
more than 25% of the time, return that integer.
'''

class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        count = 0
        
        if len(arr) < 2:
            return arr[0]
        
        for i in arr:
            if arr.count(i) / len(arr) > 0.25:
                return i
            
        return 0