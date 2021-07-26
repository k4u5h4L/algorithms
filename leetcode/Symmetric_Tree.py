'''
Symmetric Tree
Easy

Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

 

Example 1:

Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:

Input: root = [1,2,2,null,3,null,3]
Output: false
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root == None:
            return True
        
        return self.is_symmetric(root.left, root.right)
    
    def is_symmetric(self, left, right):
        if left == None or right == None:
            return left == right
        
        if left.val != right.val:
            return False
        
        return self.is_symmetric(left.left, right.right) and self.is_symmetric(left.right, right.left)
