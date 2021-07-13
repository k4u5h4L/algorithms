'''
Subarray Product Less Than K
Medium

Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.
'''

class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0
        left = 0
        right = 0
        prod = 1
        count = 0
        while right < len(nums):
            prod = prod * nums[right]
            while prod >= k:
                prod /= nums[left]
                left += 1
            count += right - left + 1
            right += 1
        return count