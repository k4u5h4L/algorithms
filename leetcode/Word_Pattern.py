'''
Word Pattern
Easy

Given a pattern and a string s, find if s follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s.

 

Example 1:

Input: pattern = "abba", s = "dog cat cat dog"
Output: true

Example 2:

Input: pattern = "abba", s = "dog cat cat fish"
Output: false

Example 3:

Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false

Example 4:

Input: pattern = "abba", s = "dog dog dog dog"
Output: false
'''

class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        pattern = [char for char in pattern]
        s = s.split(" ")
        
        if len(pattern) != len(s):
            return False
        
        memo = {}
        
        for a, b in zip(pattern, s):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
                
        memo = {}
        
        for a, b in zip(s, pattern):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
                
        return True
