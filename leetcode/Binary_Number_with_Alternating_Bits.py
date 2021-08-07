'''
Binary Number with Alternating Bits
Easy

Given a positive integer, check whether it has alternating bits: namely, if two adjacent bits will always have different values.

 

Example 1:

Input: n = 5
Output: true
Explanation: The binary representation of 5 is: 101

Example 2:

Input: n = 7
Output: false
Explanation: The binary representation of 7 is: 111.

Example 3:

Input: n = 11
Output: false
Explanation: The binary representation of 11 is: 1011.

Example 4:

Input: n = 10
Output: true
Explanation: The binary representation of 10 is: 1010.

Example 5:

Input: n = 3
Output: false
'''


class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        binary = bin(n)[2:]
        
        if len(binary) == 1:
            return True
        
        for i in range(len(binary) - 1):
            if binary[i] == binary[i+1]:
                return False
            
        return True
