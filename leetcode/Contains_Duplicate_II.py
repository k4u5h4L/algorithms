'''
Contains Duplicate II
Easy

Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
'''

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dups = {}
        for i in range(len(nums)):
            if nums[i] in dups:
                for j in range(len(nums)):
                    if i != j and nums[i] == nums[j] and abs(i-j) <= k:
                        return True
            else:
                dups[nums[i]] = i
                    
        return False