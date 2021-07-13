'''
Remove Duplicates from Sorted List
Easy

Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        cur = head
        while cur != None and cur.next != None:
            while cur.next != None and cur.val == cur.next.val:
                cur.next = cur.next.next
            cur = cur.next
        return head