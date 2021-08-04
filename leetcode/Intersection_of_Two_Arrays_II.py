'''
Intersection of Two Arrays II
Easy

Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.

 

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]

Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Explanation: [9,4] is also accepted.
'''


class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dic1 = {}
        dic2 = {}
        
        for num in nums1:
            if num in dic1:
                dic1[num] += 1
            else:
                dic1[num] = 1
                
        for num in nums2:
            if num in dic2:
                dic2[num] += 1
            else:
                dic2[num] = 1
                
        dic = {}
        
        for x in dic1:
            if x in dic2:
                dic[x] = min(dic1[x], dic2[x])
        
        res = []
        
        for key in dic.keys():
            res.extend([key] * dic[key])
            
        return res
