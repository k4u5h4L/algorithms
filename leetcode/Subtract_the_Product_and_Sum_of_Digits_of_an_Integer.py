'''
Subtract the Product and Sum of Digits of an Integer
Easy
Given an integer number n, return the difference between the product of its digits and the sum of its digits. 
'''

class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        n = [int(char) for char in str(n)]
        print(n)
        
        pdt = 1
        
        for i in n:
            pdt *= i
            
        return (pdt - sum(n))
        