'''
Sort Characters By Frequency
Medium

Given a string s, sort it in decreasing order based on the frequency of characters, and return the sorted string.

 

Example 1:

Input: s = "tree"
Output: "eert"
Explanation: 'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

Example 2:

Input: s = "cccaaa"
Output: "aaaccc"
Explanation: Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
Note that "cacaca" is incorrect, as the same characters must be together.

Example 3:

Input: s = "Aabb"
Output: "bbAa"
Explanation: "bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.
'''


class Solution:
    def frequencySort(self, s: str) -> str:
        dic = {}
        
        s = [char for char in s]
        
        for char in s:
            if char in dic:
                dic[char] += 1
            else:
                dic[char] = 1
        
        res = ""
        for key in sorted(dic, key=lambda x: dic[x], reverse=True):
            temp = key * dic[key]
            res += temp
            
        return res
