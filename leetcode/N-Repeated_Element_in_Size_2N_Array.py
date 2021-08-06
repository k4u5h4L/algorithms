'''
N-Repeated Element in Size 2N Array
Easy

You are given an integer array nums with the following properties:

    nums.length == 2 * n.
    nums contains n + 1 unique elements.
    Exactly one element of nums is repeated n times.

Return the element that is repeated n times.

 

Example 1:

Input: nums = [1,2,3,3]
Output: 3

Example 2:

Input: nums = [2,1,2,5,3,2]
Output: 2

Example 3:

Input: nums = [5,1,5,2,5,3,5,4]
Output: 5
'''


class Solution:
    def repeatedNTimes(self, nums: List[int]) -> int:
        n = len(nums) // 2
        
        dic = {}
        
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
                
        for key, val in dic.items():
            if val == n:
                return key
            
        return -1
