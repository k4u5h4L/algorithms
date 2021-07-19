'''
Length of Last Word
Easy

Given a string s consists of some words separated by spaces, return the length of the last word in the string. If the last word does not exist, return 0.

A word is a maximal substring consisting of non-space characters only.
'''

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        words = s.split(" ")
        words = [word for word in words if word != ""]
        try:
            last_word = words[-1]
            return len(last_word)
        except IndexError:
            return 0
