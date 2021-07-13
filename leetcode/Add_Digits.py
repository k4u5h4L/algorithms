'''
Add Digits
Easy

Given an integer num, repeatedly add all its digits until the result has only one digit, and return it.
'''

class Solution:
    def addDigits(self, num: int) -> int:
        sum1 = self.getSum(num)
        while sum1 > 9:
            sum1 = self.getSum(sum1)
            
        return sum1
    
    def getSum(self, num):
        sum1 = 0
        while num > 0:
            sum1 += num % 10
            num = int(num / 10)
            
        return sum1