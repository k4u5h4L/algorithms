'''
Check If All 1's Are at Least Length K Places Away
Easy

Given an array nums of 0s and 1s and an integer k, return True if all 1's are at least k places away from each other, otherwise return False.

 

Example 1:

Input: nums = [1,0,0,0,1,0,0,1], k = 2
Output: true
Explanation: Each of the 1s are at least 2 places away from each other.

Example 2:

Input: nums = [1,0,0,1,0,1], k = 2
Output: false
Explanation: The second 1 and third 1 are only one apart from each other.

Example 3:

Input: nums = [1,1,1,1,1], k = 0
Output: true

Example 4:

Input: nums = [0,1,0,1], k = 1
Output: true
'''


class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        spaces = []
        
        for i in range(len(nums)):
            if nums[i] == 1:
                spaces.append(i)
                                
        for i in range(len(spaces)-1):
            if abs(spaces[i+1] - spaces[i]) <= k:
                return False
            
        return True
