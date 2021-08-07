'''
Maximum 69 Number
Easy

Given a positive integer num consisting only of digits 6 and 9.

Return the maximum number you can get by changing at most one digit (6 becomes 9, and 9 becomes 6).

 

Example 1:

Input: num = 9669
Output: 9969
Explanation: 
Changing the first digit results in 6669.
Changing the second digit results in 9969.
Changing the third digit results in 9699.
Changing the fourth digit results in 9666. 
The maximum number is 9969.

Example 2:

Input: num = 9996
Output: 9999
Explanation: Changing the last digit 6 to 9 results in the maximum number.

Example 3:

Input: num = 9999
Output: 9999
Explanation: It is better not to apply any change.
'''

class Solution:
    def maximum69Number (self, num: int) -> int:
        num = str(num)
        
        res = ""
        
        found = False
        
        for char in num:
            if char == '6' and found == False:
                res += '9'
                found = True
            else:
                res += char
                
        return int(res)
