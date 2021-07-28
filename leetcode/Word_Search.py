'''
Word Search
Medium

Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Example 2:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Example 3:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
'''

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == word[0] and self.dfs(board, i, j, 0, word):
                    return True
        
        return False
    
    def dfs(self, board, i, j, count, word):
        if count == len(word):
            return True
        
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[i]) or board[i][j] != word[count]:
            return False
        
        temp = board[i][j]
        board[i][j] = " "
        
        found = self.dfs(board, i+1, j, count+1, word) or self.dfs(board, i-1, j, count+1, word) or self.dfs(board, i, j+1, count+1, word) or self.dfs(board, i, j-1, count+1, word)
        
        board[i][j] = temp
        return found
        
