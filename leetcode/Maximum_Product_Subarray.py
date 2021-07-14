'''
Maximum Product Subarray
Medium

Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

It is guaranteed that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.
'''

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_pdt = max(nums)
        cur_min = 1
        cur_max = 1
        
        for num in nums:
            if num == 0:
                cur_min = 1
                cur_max = 1
            else:
                temp = cur_max * num
                cur_max = max(num * cur_max, num * cur_min, num)
                cur_min = min(temp, num * cur_min, num)
                max_pdt = max(max_pdt, cur_max)
        return max_pdt