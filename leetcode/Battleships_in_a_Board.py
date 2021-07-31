'''
Battleships in a Board
Medium

Given an m x n matrix board where each cell is a battleship 'X' or empty '.', return the number of the battleships on board.

Battleships can only be placed horizontally or vertically on board. In other words, they can only be made of the shape 1 x k (1 row, k columns) or k x 1 (k rows, 1 column), where k can be of any size. At least one horizontal or vertical cell separates between two battleships (i.e., there are no adjacent battleships).

 

Example 1:

Input: board = [["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]
Output: 2

Example 2:

Input: board = [["."]]
Output: 0
'''


# Naive approach

class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        count = 0
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 'X':
                    count += self.dfs(board, i, j)
        
        return count
    
    def dfs(self, board, i, j):
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[i]) or board[i][j] != 'X':
            return 0
        
        board[i][j] = '.'
        
        self.dfs(board, i+1, j)
        self.dfs(board, i-1, j)
        self.dfs(board, i, j+1)
        self.dfs(board, i, j-1)
        
        return 1
      
      

      
# optimised approach


class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        count = 0
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == '.':
                    continue
                if i > 0 and board[i-1][j] == 'X':
                    continue
                if j > 0 and board[i][j-1] == 'X':
                    continue
                
                count += 1
        
        return count
