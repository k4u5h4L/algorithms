'''
Insertion Sort List
Medium

Given the head of a singly linked list, sort the list using insertion sort, and return the sorted list's head.

The steps of the insertion sort algorithm:

    Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
    At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
    It repeats until no input elements remain.

The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.

 

Example 1:

Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:

Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # if no elememt present
        if head == None:
            return head
        
        res = None
        
        cur = head
        
        while cur != None:
            # if result is empty, add one node and continue
            if res == None:
                res = ListNode(cur.val)
                cur = cur.next
                continue
                
            temp = ListNode(cur.val)
            cur1 = res
            prev = cur1
            
            # if only one element present, check where to insert it (back or front) and then continue
            if res.next == None:
                if res.val > temp.val:
                    temp.next = res
                    res = temp
                else:
                    res.next = temp
                    
                cur = cur.next
                continue
                
            # print(res)
            
            # if value is less than head, then no need to traverse list. just append at head and continue
            if temp.val < res.val:
                temp.next = res
                res = temp
                
                cur = cur.next
                continue
            
            # for any other case, traverse list and see the optimal place to insert new node
            while cur1 and temp.val >= cur1.val:
                prev = cur1
                cur1 = cur1.next
            
            prev.next = temp
            temp.next = cur1
            
            cur = cur.next
            
        return res
