'''
Add to Array-Form of Integer
Easy

The array-form of an integer num is an array representing its digits in left to right order.

    For example, for num = 1321, the array form is [1,3,2,1].

Given num, the array-form of an integer, and an integer k, return the array-form of the integer num + k.
'''

class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        res = [str(n) for n in num]
        res = str(int(''.join(res)) + k)
        return [int(char) for char in res]