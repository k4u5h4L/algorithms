'''
Minimum Path Sum
Medium

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

 

Example 1:

Input: grid = [
                [1,3,1],
                [1,5,1],
                [4,2,1]
              ]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.

Example 2:

Input: grid = [[1,2,3],[4,5,6]]
Output: 12
'''


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = []
        
        for i in range(len(grid)):
            dp.append([0] * len(grid[i]))
                    
        for i in range(len(dp)):
            for j in range(len(dp[i])):
                dp[i][j] += grid[i][j]
                
                if i > 0 and j > 0:
                    dp[i][j] += min(dp[i-1][j], dp[i][j-1])
                    
                elif i > 0:
                    dp[i][j] += dp[i-1][j]
                elif j > 0:
                    dp[i][j] += dp[i][j-1]
         
        return dp[-1][-1]
