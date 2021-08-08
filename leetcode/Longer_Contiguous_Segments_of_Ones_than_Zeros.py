'''
Longer Contiguous Segments of Ones than Zeros
Easy

Given a binary string s, return true if the longest contiguous segment of 1s is strictly longer than the longest contiguous segment of 0s in s. Return false otherwise.

    For example, in s = "110100010" the longest contiguous segment of 1s has length 2, and the longest contiguous segment of 0s has length 3.

Note that if there are no 0s, then the longest contiguous segment of 0s is considered to have length 0. The same applies if there are no 1s.

 

Example 1:

Input: s = "1101"
Output: true
Explanation:
The longest contiguous segment of 1s has length 2: "1101"
The longest contiguous segment of 0s has length 1: "1101"
The segment of 1s is longer, so return true.

Example 2:

Input: s = "111000"
Output: false
Explanation:
The longest contiguous segment of 1s has length 3: "111000"
The longest contiguous segment of 0s has length 3: "111000"
The segment of 1s is not longer, so return false.

Example 3:

Input: s = "110100010"
Output: false
Explanation:
The longest contiguous segment of 1s has length 2: "110100010"
The longest contiguous segment of 0s has length 3: "110100010"
The segment of 1s is not longer, so return false.
'''


class Solution:
    def checkZeroOnes(self, s: str) -> bool:
        nums = [int(char) for char in s]
                        
        if len(nums) == 1:
            return True if nums[0] == 1 else False
        
        elif 1 not in nums:
            return False
        
        elif 0 not in nums:
            return True
        
        max_len1 = 1
        cur_len1 = 0
        
        max_len0 = 1
        cur_len0 = 0
        
        for val in nums:
            if val == 1:
                cur_len1 += 1
                cur_len0 = 0
            else:
                cur_len1 = 0
                cur_len0 += 1
                
            max_len1 = max(max_len1, cur_len1)
            max_len0 = max(max_len0, cur_len0)
            
        return True if max_len1 > max_len0 else False
