'''
Top K Frequent Elements
Medium

Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:

Input: nums = [1], k = 1
Output: [1]
'''


from queue import PriorityQueue

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        def sort_func(x):
            return dic[x]
        dic = {}
        
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
                
        nums.sort(key=sort_func, reverse=True)
        
        s = set()
        res = []
        
        for num in nums:
            if len(s) == k:
                break
            if num in s:
                continue
            else:
                res.append(num)
                s.add(num)
        
        return res
