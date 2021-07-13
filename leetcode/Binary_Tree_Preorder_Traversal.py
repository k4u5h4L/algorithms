'''
 Binary Tree Preorder Traversal
Easy

Given the root of a binary tree, return the preorder traversal of its nodes' values.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        
        self.preorder(root, res)
        
        return res
    
    def preorder(self, root, res):
        if root == None:
            return
        
        res.append(root.val)
        self.preorder(root.left, res)
        self.preorder(root.right, res)