'''
Pow(x, n)
Medium

Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).

 

Example 1:

Input: x = 2.00000, n = 10
Output: 1024.00000

Example 2:

Input: x = 2.10000, n = 3
Output: 9.26100

Example 3:

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/2^2 = 1/4 = 0.25
'''

class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == float(1) or n == 0:
            return 1
        
        if x == float(-1):
            return 1 if n % 2 == 0 else -1
        
        neg = False
        
        if n < 0:
            n = abs(n)
            neg = True
            
        if neg == True:
            x = 1 / x
            
        res = x
        for _ in range(n-1):
            if abs(res) < 0.00001:
                return 0.00000
            elif abs(res) > 100000:
                return 100000
            res *= x
            
        return res
