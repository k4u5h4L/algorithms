'''
Keyboard Row
Easy

Given an array of strings words, return the words that can be typed using letters of the alphabet on only one row of American keyboard like the image below.

In the American keyboard:

    the first row consists of the characters "qwertyuiop",
    the second row consists of the characters "asdfghjkl", and
    the third row consists of the characters "zxcvbnm".

 

Example 1:

Input: words = ["Hello","Alaska","Dad","Peace"]
Output: ["Alaska","Dad"]

Example 2:

Input: words = ["omk"]
Output: []

Example 3:

Input: words = ["adsdf","sfd"]
Output: ["adsdf","sfd"]
'''


class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        
        row1 = "qwertyuiop"
        row2 = "asdfghjkl"
        row3 = "zxcvbnm"
        
        res = []
        
        for word in words:
            if word[0].lower() in row1:
                print(f"{word} in row1")
                for char in word:
                    if char.lower() not in row1:
                        break
                        
                else:
                    res.append(word)
                
            elif word[0].lower() in row2:
                print(f"{word} in row2")
                for char in word:
                    if char.lower() not in row2:
                        break
                        
                else:
                    res.append(word)
                    
            elif word[0].lower() in row3:
                print(f"{word} in row3")
                for char in word:
                    if char.lower() not in row3:
                        break
                        
                else:
                    res.append(word)
                    
        return res
