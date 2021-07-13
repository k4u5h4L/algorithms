'''
Find First and Last Position of Element in Sorted Array
Medium

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.
'''

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        res = [-1, -1]
        res[0] = self.starting(nums, target)
        res[1] = self.ending(nums, target)
        return res
    
    def starting(self, nums, target):
        index = -1
        start = 0
        end = len(nums) - 1
        
        while start <= end:
            mid = int(start + (end - start) / 2)
            if nums[mid] >= target:
                end = mid - 1
            else:
                start = mid + 1
                
            if nums[mid] == target:
                index = mid
                
        return index
    
    def ending(self, nums, target):
        index = -1
        start = 0
        end = len(nums) - 1
        
        while start <= end:
            mid = int(start + (end - start) / 2)
            if nums[mid] <= target:
                start = mid + 1
            else:
                end = mid - 1
                
            if nums[mid] == target:
                index = mid
                
        return index