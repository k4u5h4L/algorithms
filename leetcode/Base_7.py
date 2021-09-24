'''
Base 7
Easy

Given an integer num, return a string of its base 7 representation.

 

Example 1:

Input: num = 100
Output: "202"

Example 2:

Input: num = -7
Output: "-10"
'''


class Solution:
    def convertToBase7(self, num: int) -> str:
        if num == 0:
            return "0"
        
        res = ""
        flag = True
        
        if num < 0:
            flag = False
            num = abs(num)
                    
        while num > 0:
            res += str(num % 7)
            num = num // 7
        
        return res[::-1] if flag == True else f"-{res[::-1]}"
