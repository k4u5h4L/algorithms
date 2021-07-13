'''
Majority Element
Easy

Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.
'''

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        memo = {}
        for a in nums:
            if a in memo:
                memo[a] += 1
            else:
                memo[a] = 1
        max_val = 0
        max_ele = nums[0]
        for key, value in memo.items():
            if value > max_val:
                max_ele = key
                max_val = value
        return max_ele