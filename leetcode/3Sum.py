'''
3Sum
Medium

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Example 2:

Input: nums = []
Output: []

Example 3:

Input: nums = [0]
Output: []
'''



class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        
        nums.sort()
        
        res = []
        
        for i in range(len(nums)-2):
            if i == 0 or (i > 0 and nums[i] != nums[i-1]):                
                left, right = i+1, len(nums) - 1
                target = 0 - nums[i]
                
                while left < right:
                    if nums[left] + nums[right] == target:
                        res.append([nums[left], nums[right], nums[i]])
                        
                        while left < right and nums[left] == nums[left+1]:
                            left += 1
                        while left < right and nums[right] == nums[right-1]:
                            right -= 1
                            
                        left += 1
                        right -= 1
                        
                    elif nums[left] + nums[right] > target:
                        right -= 1
                    else:
                        left += 1
        
        return res
