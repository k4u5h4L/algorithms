'''
Find the Duplicate Number
Medium

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.
'''

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            if nums[abs(nums[i]) - 1] < 0:
                return abs(nums[i])
            else:
                nums[abs(nums[i]) - 1] = -nums[abs(nums[i]) - 1]
                
        return -1
#         nums.sort()
        
#         for i in range(1, len(nums)):
#             if nums[i-1] == nums[i]:
#                 return nums[i]