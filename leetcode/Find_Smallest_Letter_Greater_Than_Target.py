'''
Find Smallest Letter Greater Than Target
Easy

Given a characters array letters that is sorted in non-decreasing order and a character target, return the smallest character in the array that is larger than target.

Note that the letters wrap around.

    For example, if target == 'z' and letters == ['a', 'b'], the answer is 'a'.

 

Example 1:

Input: letters = ["c","f","j"], target = "a"
Output: "c"

Example 2:

Input: letters = ["c","f","j"], target = "c"
Output: "f"

Example 3:

Input: letters = ["c","f","j"], target = "d"
Output: "f"

Example 4:

Input: letters = ["c","f","j"], target = "g"
Output: "j"

Example 5:

Input: letters = ["c","f","j"], target = "j"
Output: "c"
'''


class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        word_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 
                       'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 
                       'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
        
        target_num = word_to_num[target]
                
        for char in letters:
            if word_to_num[char] > target_num:
                return char
        return letters[0]
            
