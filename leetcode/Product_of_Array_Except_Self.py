'''
Product of Array Except Self
Medium

Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.
'''

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        left_pdt = [1]
        
        right_pdt = nums.copy()
        right_pdt[len(nums)-1] = 1

        for i in range(len(nums)-1):
            left_pdt.append(nums[i] * left_pdt[i])
        i = len(nums) - 2

        while i >= 0:
            right_pdt[i] = nums[i+1] * right_pdt[i+1]
            i -= 1

        res = []

        for i in range(len(nums)):
            res.append(left_pdt[i] * right_pdt[i])

        return res