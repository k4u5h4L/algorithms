'''
Longest Palindromic Substring
Medium

Given a string s, return the longest palindromic substring in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:

Input: s = "cbbd"
Output: "bb"

Example 3:

Input: s = "a"
Output: "a"

Example 4:

Input: s = "ac"
Output: "a"
'''

# pretty fast solution using sliding window

class Solution:
    def longestPalindrome(self, s: str) -> str:
        def is_palin(s):
            return s == s[::-1]
        
        if len(s) == 1:
            return s
        elif len(s) == 2:
            if s[0] == s[1]:
                return s
            else:
                return s[0]
        
        res = s[0]
        
        left = 0
        right = 1
        
        while right < len(s):
            temp = s[left:right+1]
            
            if is_palin(temp):
                right += 1
                
                if left > 0:
                    left -= 1
                
                if len(res) < len(temp):
                    res = temp
                    
                continue
                    
            if left >= right:
                right += 1
            else:
                left += 1
                    
        return res


# naive way, don't use this lol

class Solution:
    def longestPalindrome(self, s: str) -> str:
        def is_palin(s):
            return s == s[::-1]
        
        if len(s) == 1:
            return s
        elif len(s) == 2:
            if s[0] == s[1]:
                return s
            else:
                return s[0]
        
        res = s[0]
        
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                temp = s[i:j+1]
                
                if len(temp) > len(res) and is_palin(temp):
                    res = s[i:j+1]
                    
        return res