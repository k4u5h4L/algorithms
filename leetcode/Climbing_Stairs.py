'''
Climbing Stairs
Easy

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
'''

import math
class Solution:
    def climbStairs(self, n: int) -> int:
        return self.total_ways(0, n, {})
        
    def total_ways(self, start, end, memo):
        if start in memo:
            return memo[start]
        if start == end:
            return 1
        elif start > end:
            return 0
        else:
            memo[start] = self.total_ways(start+1, end, memo) + self.total_ways(start+2, end, memo)
            return memo[start]