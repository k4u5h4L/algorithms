'''
Flatten Binary Tree to Linked List
Medium

Given the root of a binary tree, flatten the tree into a "linked list":

    The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
    The "linked list" should be in the same order as a pre-order traversal of the binary tree.

 

Example 1:

Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

Example 2:

Input: root = []
Output: []

Example 3:

Input: root = [0]
Output: [0]
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            return
        
        arr = []
        
        self.preorder(root, arr)
                
        for i in range(len(arr) - 1):
            root.val = arr[i]
            root.left = None
            
            if root.right == None:
                root.right = TreeNode()
                
            root = root.right
            
        if len(arr) >= 1:
            root.val = arr[-1]
        
        return
    
    def preorder(self, root, arr):
        if root == None:
            return
        arr.append(root.val)
        
        self.preorder(root.left, arr)
        self.preorder(root.right, arr)
