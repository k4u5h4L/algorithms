'''
Find Words That Can Be Formed by Characters
Easy

You are given an array of strings words and a string chars.

A string is good if it can be formed by characters from chars (each character can only be used once).

Return the sum of lengths of all good strings in words.

 

Example 1:

Input: words = ["cat","bt","hat","tree"], chars = "atach"
Output: 6
Explanation: The strings that can be formed are "cat" and "hat" so the answer is 3 + 3 = 6.

Example 2:

Input: words = ["hello","world","leetcode"], chars = "welldonehoneyr"
Output: 10
Explanation: The strings that can be formed are "hello" and "world" so the answer is 5 + 5 = 10.
'''


class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:  
        res = []
        
        for word in words:
            temp = word
            
            for c in chars:
                temp = temp.replace(c, "", 1)
                
                if temp == "":
                    res.append(word)
                    break
                                        
        length = 0
        
        for word in res:
            length += len(word)
            
        return length
