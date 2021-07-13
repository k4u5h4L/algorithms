'''
Single Number
Easy

Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.
'''

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        memo  ={}
        for i in nums:
            if i in memo:
                memo[i] += 1
            else:
                memo[i] = 1
        for i in nums:
            if i in memo and memo[i] == 1:
                return i
        return -1