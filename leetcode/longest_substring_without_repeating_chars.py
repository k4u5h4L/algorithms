"""
Longest Substring Without Repeating Characters
Medium

Given a string s, find the length of the longest substring without repeating characters.
"""

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left = 0
        right = 0
        
        max_len = 0
        memo = {}
        
        while right < len(s):
            if s[right] in memo:
                memo.pop(s[left])
                left += 1
                continue
            
            memo[s[right]] = True
            max_len = max(max_len, len(s[left:right+1]))
            right += 1
            
            
        return max_len;