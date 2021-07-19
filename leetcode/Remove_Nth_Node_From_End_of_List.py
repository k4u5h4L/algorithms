'''
Remove Nth Node From End of List
Medium

Given the head of a linked list, remove the nth node from the end of the list and return its head.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        temp = ListNode()
        temp.next = head
        
        slow = temp
        fast = temp
        
        for _ in range(n+1):
            fast = fast.next
        
        while fast != None:
            fast = fast.next
            slow = slow.next
        
        slow.next = slow.next.next
        return temp.next
