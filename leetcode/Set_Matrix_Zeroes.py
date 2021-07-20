'''
Set Matrix Zeroes
Medium

Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's, and return the matrix.

You must do it in place.

 

Example 1:

Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:

Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
'''

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        res = []
        for i in range(len(matrix)):
            res.append(matrix[i].copy())
        
        for i in range(len(res)):
            for j in range(len(res[i])):
                if res[i][j] == 0:
                    for k in range(len(res)):
                        matrix[k][j] = 0
                    for k in range(len(res[i])):
                        matrix[i][k] = 0
