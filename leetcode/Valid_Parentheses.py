'''
Valid Parentheses
Easy

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.

'''

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        chars = [char for char in s]
        
        for c in chars:
            if c == '(' or c == '[' or c == '{':
                stack.append(c)
            elif stack and c == ')' and stack[-1:][0] == '(':
                stack.pop()
            elif stack and c == ']' and stack[-1:][0] == '[':
                stack.pop()
            elif stack and c == '}' and stack[-1:][0] == '{':
                stack.pop()
            else:
                return False
        return True if len(stack) == 0 else False
