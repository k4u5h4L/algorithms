'''
Consecutive Characters
Easy

Given a string s, the power of the string is the maximum length of a non-empty substring that contains only one unique character.

Return the power of the string.

 

Example 1:

Input: s = "leetcode"
Output: 2
Explanation: The substring "ee" is of length 2 with the character 'e' only.

Example 2:

Input: s = "abbcccddddeeeeedcba"
Output: 5
Explanation: The substring "eeeee" is of length 5 with the character 'e' only.

Example 3:

Input: s = "triplepillooooow"
Output: 5

Example 4:

Input: s = "hooraaaaaaaaaaay"
Output: 11

Example 5:

Input: s = "tourist"
Output: 1
'''


class Solution:
    def maxPower(self, s: str) -> int:
        if len(s) == 1:
            return 1
        
        max_len = 1
        
        left = 0
        right = 0
        
        while right < len(s):
            cache = s[right]
            
            right += 1
            
            if right < len(s) and s[right] == cache:
                max_len = max(max_len, right - left + 1)
                continue
            else:
                left = right
                
        return max_len
