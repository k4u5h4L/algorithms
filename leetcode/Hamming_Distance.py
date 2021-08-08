'''
Hamming Distance
Easy

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers x and y, return the Hamming distance between them.

 

Example 1:

Input: x = 1, y = 4
Output: 2
Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
The above arrows point to positions where the corresponding bits are different.

Example 2:

Input: x = 3, y = 1
Output: 1
'''


class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        bits1 = bin(x)[2:]
        bits2 = bin(y)[2:]
        
        dist = 0
        
        if len(bits1) < len(bits2):
            temp = len(bits2) - len(bits1)
            
            bits1 = ("0" * temp) + bits1
            
        elif len(bits2) < len(bits1):
            temp = len(bits1) - len(bits2)
            
            bits2 = ("0" * temp) + bits2
        
        for b1, b2 in zip(list(bits1), list(bits2)):
            if b1 != b2:
                dist += 1
                
        return dist
