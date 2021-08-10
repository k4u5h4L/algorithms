'''
Factorial Trailing Zeroes
Easy

Given an integer n, return the number of trailing zeroes in n!.

Follow up: Could you write a solution that works in logarithmic time complexity?

 

Example 1:

Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.

Example 2:

Input: n = 5
Output: 1
Explanation: 5! = 120, one trailing zero.

Example 3:

Input: n = 0
Output: 0
'''


class Solution:
    def trailingZeroes(self, n: int) -> int:
        fact = 1
        
        while n > 0:
            fact *= n
            n -= 1
            
        fact = str(fact)[::-1]
        res = 0
        
        for char in fact:
            if char != '0':
                return res
            
            res += 1
        
        return res
