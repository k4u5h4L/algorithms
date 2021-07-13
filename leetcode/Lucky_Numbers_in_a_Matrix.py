'''
Lucky Numbers in a Matrix
Easy

Given a m * n matrix of distinct numbers, return all lucky numbers in the matrix in any order.

A lucky number is an element of the matrix such that it is the minimum element in its row and maximum in its column.
'''

class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        res = []
        
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if self.isLucky(matrix[i][j], matrix[i], self.getCols(j, matrix), matrix):
                    res.append(matrix[i][j])
                    
        return res
                    
    def getCols(self, column, mat):
        col = []
        
        for i in range(len(mat)):
            col.append(mat[i][column])
            
        return col
                
    
    def isLucky(self, num, row, col, mat):
        if num == min(row) and num == max(col):
            return True
        else:
            return False