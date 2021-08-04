'''
Find Common Characters
Easy

Given a string array words, return an array of all characters that show up in all strings within the words (including duplicates). You may return the answer in any order.

 

Example 1:

Input: words = ["bella","label","roller"]
Output: ["e","l","l"]

Example 2:

Input: words = ["cool","lock","cook"]
Output: ["c","o"]
'''


class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        if len(words) == 1:
            return [char for char in words[0]]
        
        dic = {}
        
        for char in words[0]:
            if char in dic:
                dic[char] += 1
            else:
                dic[char] = 1
                
        words = words[1:]
        
        for word in words:
            temp_dic = dic.copy()
            dic = {}
            for char in word:
                if char in temp_dic and temp_dic[char] > 0:
                    temp_dic[char] -= 1
                    if char in dic:
                        dic[char] += 1
                    else:
                        dic[char] = 1
                   
        res = []
        for key, value in dic.items():
            res.extend([key] * value)
            
        return res
