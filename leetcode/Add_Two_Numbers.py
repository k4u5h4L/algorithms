'''
Add Two Numbers
Medium

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

Example 1:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        
        digit1 = ""
        cur1 = l1
        
        while cur1 != None:
            digit1 += str(cur1.val)
            cur1 = cur1.next
            
        digit2 = ""
        cur2 = l2
        
        while cur2 != None:
            digit2 += str(cur2.val)
            cur2 = cur2.next
            
        digit1 = digit1[::-1]
        digit2 = digit2[::-1]
            
        digit1 = int(digit1)
        digit2 = int(digit2)
        
        res = str(digit1 + digit2)
        res = res[::-1]
        
        head = ListNode(int(res[0]))
        res = res[1:]
        cur = head
        
        for char in res:
            cur.next = ListNode(int(char))
            cur = cur.next
            
        return head
