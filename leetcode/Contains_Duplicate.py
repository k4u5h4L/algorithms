'''
Contains Duplicate
Easy

Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
'''

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        dups = {}
        for num in nums:
            if num in dups:
                return True
            else:
                dups[num] = True
        return False