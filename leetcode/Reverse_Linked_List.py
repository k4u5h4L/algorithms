'''
Reverse Linked List
Easy

Given the head of a singly linked list, reverse the list, and return the reversed list.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        
        while (head != None):
            lnext = head.next
            head.next = prev
            prev = head
            head = lnext
            
        return prev