'''
Rotate List
Medium

Given the head of a linked list, rotate the list to the right by k places.

 

Example 1:

Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Example 2:

Input: head = [0,1,2], k = 4
Output: [2,0,1]
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head == None or head.next == None:
            return head
        size = 0
        cur = head
        
        # to find the size of the list to reduce unnecessary roatations
        while cur != None:
            size += 1
            cur = cur.next
        
        # running the loop for only required size which is (k mod size)
        for _ in range(k % size):
            fast = head
            prev = fast
            while fast.next != None:
                prev = fast
                fast = fast.next
            prev.next = None
            fast.next = head
            head = fast
        return head
