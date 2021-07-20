'''
Rotting Oranges
Medium

You are given an m x n grid where each cell can have one of three values:

    0 representing an empty cell,
    1 representing a fresh orange, or
    2 representing a rotten orange.

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.

 

Example 1:

Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4

Example 2:

Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.

Example 3:

Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
'''

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        temp = []
        minutes = 0
        for i in range(len(grid)):
            temp.append(grid[i].copy())
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 2:
                    if i > 0 and temp[i-1][j] == 1:
                        temp[i-1][j] = 2
                    if i < len(grid) - 1 and temp[i+1][j] == 1:
                        temp[i+1][j] = 2
                    
                    if j > 0 and temp[i][j-1] == 1:
                        temp[i][j-1] = 2
                    if j < len(grid[i]) - 1 and temp[i][j+1] == 1:
                        temp[i][j+1] = 2
        minutes += 1
        
        while temp != grid:
            grid = []
            for i in range(len(temp)):
                grid.append(temp[i].copy())
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == 2:
                        if i > 0 and temp[i-1][j] == 1:
                            temp[i-1][j] = 2
                        if i < len(grid) - 1 and temp[i+1][j] == 1:
                            temp[i+1][j] = 2

                        if j > 0 and temp[i][j-1] == 1:
                            temp[i][j-1] = 2
                        if j < len(grid[i]) - 1 and temp[i][j+1] == 1:
                            temp[i][j+1] = 2
            minutes += 1
            
        print(temp)
            
        for i in range(len(temp)):
            if temp[i].count(1) > 0:
                return -1
            
        return minutes - 1
