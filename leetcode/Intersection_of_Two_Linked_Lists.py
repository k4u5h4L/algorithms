'''
Intersection of Two Linked Lists
Easy

Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:

a1 -> a2 -------↓
                -> c1 -> c2 -> c3
b1 -> b2 -> b3 -↑

It is guaranteed that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        cur_a = headA
        cur_b = headB
        
        while cur_a != cur_b:
            if cur_a == None:
                cur_a = headB
            else:
                cur_a = cur_a.next
                
            if cur_b == None:
                cur_b = headA
            else:
                cur_b = cur_b.next
        
        return cur_a
