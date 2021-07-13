'''
Longest Common Prefix
Easy

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".
'''


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        min_len = len(strs[0])
        for word in strs:
            min_len = min(min_len, len(word))
        res = ""
        for ind in range(len(strs)):
            for i in range(min_len):
                if self.isEqual(strs, i):
                    res = strs[ind][:i+1]
        return res
    
    def isEqual(self, strs, index):
        for i in range(1, len(strs)):
            if strs[i-1][:index+1] != strs[i][:index+1]:
                return False
        return True