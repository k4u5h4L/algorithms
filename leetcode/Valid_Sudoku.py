'''
Valid Sudoku
Medium

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

    Each row must contain the digits 1-9 without repetition.
    Each column must contain the digits 1-9 without repetition.
    Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Note:

    A Sudoku board (partially filled) could be valid but is not necessarily solvable.
    Only the filled cells need to be validated according to the mentioned rules.

'''

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        for i in range(0, 9):
            for j in range(0, 9):
                if not self.isValid(board, i, j):
                    return False
        return True
    
    def isValid(self, arr, row, col):
        return (self.notInRow(arr, row) and self.notInCol(arr, col) and
            self.notInBox(arr, row - row % 3, col - col % 3))
    
    def notInBox(self, arr, startRow, startCol):
        st = set()
        for row in range(0, 3):
            for col in range(0, 3):
                curr = arr[row + startRow][col + startCol]
                if curr in st:
                    return False
                if curr != '.':
                    st.add(curr)
        return True
    
    def notInCol(self, arr, col):
        st = set()
        for i in range(0, 9):
            if arr[i][col] in st:
                return False
            if arr[i][col] != '.':
                st.add(arr[i][col])

        return True
    
    def notInRow(self, arr, row):
        st = set()

        for i in range(0, 9):
            if arr[row][i] in st:
                return False
            if arr[row][i] != '.':
                st.add(arr[row][i])

        return True