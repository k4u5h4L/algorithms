'''
Power of Three
Easy

Given an integer n, return true if it is a power of three. Otherwise, return false.

An integer n is a power of three, if there exists an integer x such that n == 3x.
'''

class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1:
            return True
        if n < 3:
            return False
        while n > 1:
            if n % 3 != 0:
                return False
            n = int(n / 3)
        return True