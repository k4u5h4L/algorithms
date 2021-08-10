'''
Is Subsequence
Easy

Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

 

Example 1:

Input: s = "abc", t = "ahbgdc"
Output: true

Example 2:

Input: s = "axc", t = "ahbgdc"
Output: false
'''


class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        for char in s:
            while i < len(t) and t[i] != char:
                i += 1
                
            if i >= len(t):
                return False
            elif t[i] == char:
                i += 1
                continue
                
        return True
