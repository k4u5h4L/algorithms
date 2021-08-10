'''
Ransom Note
Easy

Given two stings ransomNote and magazine, return true if ransomNote can be constructed from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.

 

Example 1:

Input: ransomNote = "a", magazine = "b"
Output: false

Example 2:

Input: ransomNote = "aa", magazine = "ab"
Output: false

Example 3:

Input: ransomNote = "aa", magazine = "aab"
Output: true
'''

class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:        
        for char in ransomNote:
            if char not in magazine:
                return False
            
            magazine = magazine.replace(char, "", 1)
            
        return True
