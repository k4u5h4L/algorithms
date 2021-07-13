'''
Move Zeroes
Easy

Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.
'''

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        num_zeros = nums.count(0)
        for _ in range(num_zeros):
            nums.remove(0)
        for _ in range(num_zeros):
            nums.append(0)