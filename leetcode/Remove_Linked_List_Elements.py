'''
Remove Linked List Elements
Easy

Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        cur = head
        
        while cur != None and cur.next != None:
            while cur.next != None and cur.next.val == val:
                cur.next = cur.next.next
            cur = cur.next
            
        if head != None and head.val == val:
            head = head.next
            return head
        return head