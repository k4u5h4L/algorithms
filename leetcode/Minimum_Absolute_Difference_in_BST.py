'''
Minimum Absolute Difference in BST
Easy

Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.

 

Example 1:

Input: root = [4,2,6,1,3]
Output: 1

Example 2:

Input: root = [1,0,48,null,null,12,49]
Output: 1
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        vals = []
        
        self.inorder(root, vals)
        
        res = max(vals)
                
        for i in range(len(vals)-1):
            res = min(res, abs(vals[i] - vals[i+1]))
                
        return res
    
    def inorder(self, root, vals):
        if root == None:
            return
        
        self.inorder(root.left, vals)
        vals.append(root.val)
        self.inorder(root.right, vals)
