'''
Add Binary
Easy

Given two binary strings a and b, return their sum as a binary string.
'''

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        res = int(a, 2) + int(b, 2)
        
        return bin(res)[2:]