'''
Isomorphic Strings
Easy

Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

 

Example 1:

Input: s = "egg", t = "add"
Output: true

Example 2:

Input: s = "foo", t = "bar"
Output: false

Example 3:

Input: s = "paper", t = "title"
Output: true
'''

class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        memo = {}
        
        if len(s) != len(t):
            return False
        
        for a, b in zip(s, t):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
        
        memo = {}
        
        for a, b in zip(t, s):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
        
        return True
