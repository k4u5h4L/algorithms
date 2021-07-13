'''
Search in a Binary Search Tree
Easy

You are given the root of a binary search tree (BST) and an integer val.

Find the node in the BST that the node's value equals val and return the subtree rooted with that node. If such a node does not exist, return null.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        return self.searchNode(root, val)
        
    def searchNode(self, root, val):
        if root == None:
            return None
        
        if val == root.val:
            return root
        elif val < root.val:
            return self.searchNode(root.left, val)
        else:
            return self.searchNode(root.right, val)