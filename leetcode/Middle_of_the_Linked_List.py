'''
Middle of the Linked List
Easy

Given a non-empty, singly linked list with head node head, return a middle node of linked list.

If there are two middle nodes, return the second middle node.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            
        return slow