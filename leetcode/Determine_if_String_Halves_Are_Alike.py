'''
Determine if String Halves Are Alike
Easy

You are given a string s of even length. Split this string into two halves of equal lengths, and let a be the first half and b be the second half.

Two strings are alike if they have the same number of vowels ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'). Notice that s contains uppercase and lowercase letters.

Return true if a and b are alike. Otherwise, return false.
'''

class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        s1 = s[:int(len(s) / 2)]
        s2 = s[int(len(s) / 2):]
        
        c1 = s1.count("a")
        c1 += s1.count("e")
        c1 += s1.count("i")
        c1 += s1.count("o")
        c1 += s1.count("u")
        
        c1 += s1.count("A")
        c1 += s1.count("E")
        c1 += s1.count("I")
        c1 += s1.count("O")
        c1 += s1.count("U")
        
        c2 = s2.count("a")
        c2 += s2.count("e")
        c2 += s2.count("i")
        c2 += s2.count("o")
        c2 += s2.count("u")
                       
        c2 +=s2.count("A")
        c2 += s2.count("E")
        c2 += s2.count("I")
        c2 += s2.count("O")
        c2 += s2.count("U")
        
        return (True if c1 == c2 else False)