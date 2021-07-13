'''
Odd Even Linked List
Medium

Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

The first node is considered odd, and the second node is even, and so on.

Note that the relative order inside both the even and odd groups should remain as it was in the input.

You must solve the problem in O(1) extra space complexity and O(n) time complexity.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        odd = head
        even = head.next
        cur = head
        
        while cur != None and cur.next != None:
            lnext = cur.next
            cur.next = cur.next.next
            cur = lnext
        while odd.next != None:
            odd = odd.next
        odd.next = even
        return head