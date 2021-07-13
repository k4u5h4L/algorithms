'''
Counting Bits
Easy

Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
'''

class Solution:
    def countBits(self, n: int) -> List[int]:
        res = []
        for i in range(n+1):
            bits = self.getBin(i)
            res.append(bits.count(1))
        return res
    
    def getBin(self, decimal):
        result = []
        while decimal > 0:
            if decimal % 2 == 0:
                result.append(0)
            else:
                result.append(1)
            decimal = int(decimal / 2)

        result.reverse()
        return result