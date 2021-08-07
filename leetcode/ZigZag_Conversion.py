'''
ZigZag Conversion
Medium

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R

And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);

 

Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I

Example 3:

Input: s = "A", numRows = 1
Output: "A"
'''


class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if len(s) == 1 or numRows == 1:
            return s
            
        zig_zag = []
        
        for _ in range(numRows):
            zig_zag.append([" "] * (len(s)))
            
        i = 0
        j = 0
        point = 0
        
        down_or_slant = True
        
        while point < len(s):
            if down_or_slant:
                while i < numRows and point < len(s):
                    zig_zag[i][j] = s[point]
                    i += 1
                    point += 1
                    
                down_or_slant = not down_or_slant
                j += 1
                
            else:
                i -= 2
                
                while i >= 0 and point < len(s):
                    zig_zag[i][j] = s[point]
                    i -= 1
                    j += 1

                    point += 1
                    
                i += 2
                
                down_or_slant = not down_or_slant
                
        res = ""
        
        for i in range(len(zig_zag)):
            for j in range(len(zig_zag[i])):
                if zig_zag[i][j] != " ":
                    res += zig_zag[i][j]

        return res
