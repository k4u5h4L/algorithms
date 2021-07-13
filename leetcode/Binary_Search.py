'''
Binary Search
Easy

Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.
'''

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        return self.binary(nums, target, 0, len(nums) - 1)
    
    def binary(self, a, key, low, high):
        if low <= high:
            mid = int((low + high) / 2)
            
            if key == a[mid]:
                return mid
            
            elif key < a[mid]:
                return self.binary(a, key, low, mid - 1)
                
            else:
                return self.binary(a, key, mid + 1, high)
        else:
            return -1