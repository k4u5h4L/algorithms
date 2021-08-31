'''
Longest Harmonious Subsequence
Easy

We define a harmonious array as an array where the difference between its maximum value and its minimum value is exactly 1.

Given an integer array nums, return the length of its longest harmonious subsequence among all its possible subsequences.

A subsequence of array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.

 

Example 1:

Input: nums = [1,3,2,2,5,2,3,7]
Output: 5
Explanation: The longest harmonious subsequence is [3,2,2,2,3].

Example 2:

Input: nums = [1,2,3,4]
Output: 2

Example 3:

Input: nums = [1,1,1,1]
Output: 0
'''


class Solution:
    def findLHS(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        
        dic = {}
        res = 0
        
        for n in nums:
            if n in dic:
                dic[n] +=1
            else:
                dic[n] = 1
                
        for key, value in dic.items():
            if key + 1 in dic:
                res = max(res, dic[key] + dic[key + 1])
                
        return res
