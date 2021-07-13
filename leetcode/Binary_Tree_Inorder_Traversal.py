'''
Binary Tree Inorder Traversal
Easy

Given the root of a binary tree, return the inorder traversal of its nodes' values.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        
        self.inorder(root, res)
        
        return res
    
    def inorder(self, root, res):
        if root == None:
            return
        
        self.inorder(root.left, res)
        res.append(root.val)
        self.inorder(root.right, res)