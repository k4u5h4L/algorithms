'''
Sort Array By Parity
Easy

Given an array nums of non-negative integers, return an array consisting of all the even elements of nums, followed by all the odd elements of nums.

You may return any answer array that satisfies this condition.
'''

class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        res = []
        
        for i in A:
            if i % 2 == 0:
                res.append(i)
                
        for i in A:
            if i % 2 == 1:
                res.append(i)
                
        return res