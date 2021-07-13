'''
First Unique Character in a String
Easy

Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.
'''

class Solution:
    def firstUniqChar(self, s: str) -> int:
        words = {}
        
        for char in s:
            try:
                words[char] = words[char] + 1
            except Exception:
                words.update({char: 1})
                        
        for (key, value) in words.items():
            if value == 1:
                return s.index(key)
            
        return -1