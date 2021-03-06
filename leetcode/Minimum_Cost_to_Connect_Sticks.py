'''
Minimim Cost to Connect Sticks
Medium

You have some sticks with positive integer lengths

You can connect any two sticks of lengths X andc Y by paying a cost X + Y.
You can perform this action until there is one stick remaining.

Return the minimum cost of connecting all the given sticks into one stick in this way.

Example 1:
Input: sticks = [2,4,3]
Output: 14

Example 2:
Input: sticks = [1,8,3,5]
Output: 30
'''

from queue import PriorityQueue

class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        cost = 0
        q = PriorityQueue()
        
        for stick in sticks:
            q.put(stick)
        
        while q.qsize() > 1:
            cur_sum = q.get() + q.get()
            cost += cur_sum
            q.put(cur_sum)
            
        return cost
