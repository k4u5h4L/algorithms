'''
Validate Binary Search Tree
Medium

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.isValid(root, None, None)
    
    def isValid(self, root, max_val, min_val):
        if root == None:
            return True
        elif max_val != None and root.val >= max_val or min_val != None and root.val <= min_val:
            return False
        else:
            return self.isValid(root.left, root.val, min_val) and self.isValid(root.right, max_val, root.val) 
