'''
Convert Binary Number in a Linked List to Integer
Easy

Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        binary = []
        while head != None:
            binary.append(head.val)
            head = head.next
            
        deci = 0
        reversedBin = [bit for bit in reversed(binary)]
        
        for i in range(len(reversedBin)):
            if reversedBin[i] == 1:
                deci += (2 ** i)
                
        return deci