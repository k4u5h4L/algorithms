'''
Search a 2D Matrix
Medium

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the previous row.

'''

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        last_ind = len(matrix[0]) - 1
        row = -1
        for i in range(len(matrix)):
            if target <= matrix[i][last_ind]:
                row = i
                break
        if row == -1:
            return False
        for ele in matrix[row]:
            if ele == target:
                return True
        return False