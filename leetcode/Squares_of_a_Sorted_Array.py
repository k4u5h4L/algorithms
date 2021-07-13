'''
Squares of a Sorted Array
Easy

Given an integer array nums sorted in non-decreasing order, 
return an array of the squares of each number sorted in non-decreasing order.
'''

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        negs = []
        pos = []
        for i in nums:
            if i < 0:
                negs.insert(0, i ** 2)
            else:
                pos.append(i ** 2)
        res = []
        l1 = 0
        l2 = 0
        while l1 < len(negs) and l2 < len(pos):
            if negs[l1] < pos[l2]:
                res.append(negs[l1])
                l1 += 1
            else:
                res.append(pos[l2])
                l2 += 1
        while l1 < len(negs):
            res.append(negs[l1])
            l1 += 1
        while l2 < len(pos):
            res.append(pos[l2])
            l2 += 1
        return res