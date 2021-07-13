'''
Unique Paths
Medium

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
'''

class Solution:
    def uniquePaths(self, m: int, n: int, memo={}) -> int:
        key = f'{m},{n}'
        if key in memo:
            return memo[key]
        if m == 1 and n == 1:
            return 1
        if m == 0 or n == 0:
            return 0
        memo[key] = self.uniquePaths(m-1, n, memo) + self.uniquePaths(m, n-1, memo)
        return memo[key]