'''
Number of Segments in a String
Easy

You are given a string s, return the number of segments in the string. 

A segment is defined to be a contiguous sequence of non-space characters.

 

Example 1:

Input: s = "Hello, my name is John"
Output: 5
Explanation: The five segments are ["Hello,", "my", "name", "is", "John"]

Example 2:

Input: s = "Hello"
Output: 1

Example 3:

Input: s = "love live! mu'sic forever"
Output: 4

Example 4:

Input: s = ""
Output: 0
'''


class Solution:
    def countSegments(self, s: str) -> int:
        if s == "":
            return 0
        
        return len([char for char in s.split(" ") if char != ""])
