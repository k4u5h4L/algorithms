'''
Backspace String Compare
Easy

Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.

Note that after backspacing an empty text, the text will continue empty.
'''

class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        s = [char for char in s]
        t = [char for char in t]
        stack_s = []
        stack_t = []
        
        for i in s:
            if i == '#':
                if len(stack_s) == 0:
                    continue
                stack_s.pop()
            else:
                stack_s.append(i)
        for i in t:
            if i == '#':
                if len(stack_t) == 0:
                    continue
                stack_t.pop()
            else:
                stack_t.append(i)
                
        return stack_s == stack_t