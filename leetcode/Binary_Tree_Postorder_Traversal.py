'''
Binary Tree Postorder Traversal
Easy

Given the root of a binary tree, return the postorder traversal of its nodes' values.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        
        self.postorder(root, res)
        
        return res
    
    def postorder(self, root, res):
        if root == None:
            return
        
        self.postorder(root.left, res)
        self.postorder(root.right, res)
        res.append(root.val)