'''
Max Consecutive Ones
Easy

Given a binary array nums, return the maximum number of consecutive 1's in the array.

 

Example 1:

Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.

Example 2:

Input: nums = [1,0,1,1,0,1]
Output: 2
'''


class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
                
        elif 1 not in nums:
            return 0
        
        max_len = 1
        cur_len = 0
        
        for val in nums:
            if val == 1:
                cur_len += 1
            else:
                cur_len = 0
                
            max_len = max(max_len, cur_len)
            
        return max_len
