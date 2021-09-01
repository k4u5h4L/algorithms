'''
Largest Substring Between Two Equal Characters
Easy

Given a string s, return the length of the longest substring between two equal characters, excluding the two characters. If there is no such substring return -1.

A substring is a contiguous sequence of characters within a string.

 

Example 1:

Input: s = "aa"
Output: 0
Explanation: The optimal substring here is an empty substring between the two 'a's.

Example 2:

Input: s = "abca"
Output: 2
Explanation: The optimal substring here is "bc".

Example 3:

Input: s = "cbzxy"
Output: -1
Explanation: There are no characters that appear twice in s.

Example 4:

Input: s = "cabbac"
Output: 4
Explanation: The optimal substring here is "abba". Other non-optimal substrings include "bb" and "".
'''


class Solution:
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        res = -1
        
        if len(s) == 1:
            return res
        
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    res = max(res, len(s[i+1:j]))
                    
        return res
