'''
First Missing Positive
Hard

Given an unsorted integer array nums, find the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses constant extra space.
'''

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        min_pos = 1
        memo = {}
        for i, a in enumerate(nums):
            memo[a] = True
            if a > 0 and a == min_pos:
                min_pos += 1
        while min_pos in memo:
            min_pos += 1
        return min_pos