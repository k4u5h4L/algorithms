'''
Maximum Product of Two Elements in an Array
Easy
Given the array of integers nums, you will choose two different indices i and j of that array. Return the maximum value of (nums[i]-1)*(nums[j]-1). 
'''

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxEle = max(nums)
        nums.pop(nums.index(maxEle))
        secMax = max(nums)
        
        return (maxEle-1) * (secMax - 1)