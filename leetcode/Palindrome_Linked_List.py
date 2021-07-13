'''
Palindrome Linked List
Easy

Given the head of a singly linked list, return true if it is a palindrome.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        slow = head
        fast = head
        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
        slow = self.reverse_list(slow)
        fast = head
        while slow != None:
            if slow.val != fast.val:
                return False
            slow = slow.next
            fast = fast.next
        return True
            
    def reverse_list(self, head):
        prev = None
        while head != None:
            lnext = head.next
            head.next = prev
            prev = head
            head = lnext
        return prev