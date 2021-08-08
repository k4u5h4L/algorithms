'''
Perfect Number
Easy

A perfect number is a positive integer that is equal to the sum of its positive divisors, excluding the number itself. A divisor of an integer x is an integer that can divide x evenly.

Given an integer n, return true if n is a perfect number, otherwise return false.

 

Example 1:

Input: num = 28
Output: true
Explanation: 28 = 1 + 2 + 4 + 7 + 14
1, 2, 4, 7, and 14 are all divisors of 28.

Example 2:

Input: num = 6
Output: true

Example 3:

Input: num = 496
Output: true

Example 4:

Input: num = 8128
Output: true

Example 5:

Input: num = 2
Output: false
'''

# [accepted] optimal solution

class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num <= 0:
            return False
        
        div_sum = 0
        i = 1
        
        while (i * i) <= num:
            if num % i == 0:
                div_sum += i
                
                if (i * i) != num:
                    div_sum += num // i
            
            i += 1
            
        return div_sum - num == num


# [accepted] very simple solution, but need to know what you're doing and not exactly an "algorithm" 

class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        return num in (6, 28, 496, 8128, 33550336)
      
      
 # [time limit exceeded] brute force solution, may slow down for large numbers

class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        divisors_sum = 0
        
        for i in range(1, num):
            if num % i == 0:
                divisors_sum += i
                
        return divisors_sum == num
