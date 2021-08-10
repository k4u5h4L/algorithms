'''
Valid Perfect Square
Easy

Given a positive integer num, write a function which returns True if num is a perfect square else False.

Follow up: Do not use any built-in library function such as sqrt.

 

Example 1:

Input: num = 16
Output: true

Example 2:

Input: num = 14
Output: false
'''


class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num == 1:
            return True
        
        i = 2
        
        while (i * i) <= num:
            if (i * i ) == num:
                return True
            
            i += 1
            
        return False
