'''
Longest Palindrome
Easy

Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.

Letters are case sensitive, for example, "Aa" is not considered a palindrome here.

 

Example 1:

Input: s = "abccccdd"
Output: 7
Explanation:
One longest palindrome that can be built is "dccaccd", whose length is 7.

Example 2:

Input: s = "a"
Output: 1

Example 3:

Input: s = "bb"
Output: 2
'''


class Solution:
    def longestPalindrome(self, s: str) -> int:        
        if len(s) == 1:
            return 1
        
        dic = {}
    
        for char in s:
            if char in dic:
                dic[char] += 1
            else:
                dic[char] = 1
                
        max_len = 0
        
        for val in dic.values():
            max_len += (val // 2) * 2
            
            if max_len % 2 == 0 and val % 2 == 1:
                max_len += 1
                
        return max_len
