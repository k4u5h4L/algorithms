'''
Integer Replacement
Medium

Given a positive integer n, you can apply one of the following operations:

    If n is even, replace n with n / 2.
    If n is odd, replace n with either n + 1 or n - 1.

Return the minimum number of operations needed for n to become 1.

 

Example 1:

Input: n = 8
Output: 3
Explanation: 8 -> 4 -> 2 -> 1

Example 2:

Input: n = 7
Output: 4
Explanation: 7 -> 8 -> 4 -> 2 -> 1
or 7 -> 6 -> 3 -> 2 -> 1

Example 3:

Input: n = 4
Output: 2
'''


class Solution:
    def integerReplacement(self, n: int) -> int:
        return self.num_steps(n) - 1
    
    def num_steps(self, n, memo={}):
        if n in memo:
            return memo[n]
        
        if n == 1:
            return 1
        
        elif n % 2 == 0:
            memo[n] = 1 + self.num_steps(n // 2, memo)
            
        else:
            memo[n] = 1 + min(self.num_steps(n + 1, memo), self.num_steps(n - 1, memo))
        
        return memo[n]
