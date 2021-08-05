'''
Fibonacci Number
Easy

The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,

F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.

Given n, calculate F(n).
'''

# bottom up approach (DP)
class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        elif n < 2:
            return 1
            
        dp = [0] * (n + 1)
    
        dp[0] = 0
        dp[1] = 1
        dp[2] = 1

        if n < 3:
            return dp[n]

        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]
    
    
 

# recursive and memoized solution (DP)
class Solution:
    def fib(self, n: int, memo={}) -> int:
        if n == 0:
            return 0
        if n <= 2:
            return 1
        if n in memo:
            return memo[n]
        else:
            memo[n] = self.fib(n-1, memo) + self.fib(n-2, memo)
            return memo[n]
