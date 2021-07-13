'''
Merge Two Sorted Lists
Easy

Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        res = None

        while l1 != None and l2 != None:
            if l1.val < l2.val:
                if res == None:
                    res = ListNode(l1.val)
                else:
                    cur = res
                    while cur != None and cur.next != None:
                        cur = cur.next

                    cur.next = ListNode(val=l1.val)
                l1 = l1.next
            else:
                if res == None:
                    res = ListNode(l2.val)
                else:
                    cur = res
                    while cur != None and cur.next != None:
                        cur = cur.next

                    cur.next = ListNode(val=l2.val)
                l2 = l2.next
        
        while l1 != None:
            if res == None:
                res = ListNode(l1.val)
                l1 = l1.next
                continue
            cur = res
            while cur != None and cur.next != None:
                cur = cur.next
                    
            cur.next = ListNode(val=l1.val)
            l1 = l1.next
        while l2 != None:
            if res == None:
                res = ListNode(l2.val)
                l2 = l2.next
                continue
            cur = res
            while cur != None and cur.next != None:
                cur = cur.next
                    
            cur.next = ListNode(val=l2.val)
            l2 = l2.next
            
        return res