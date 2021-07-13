'''
Find All Duplicates in an Array
Medium

Given an integer array nums of length n where all the integers of nums are in the range [1, n] and each integer appears once or twice, return an array of all the integers that appears twice.

You must write an algorithm that runs in O(n) time and uses only constant extra space.
'''

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        dic = {}
        res = []
        for i in nums:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
        for i in nums:
            if i in dic and dic[i] > 1:
                res.append(i)
                dic.pop(i)
        return res