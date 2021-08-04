'''
Balanced Binary Tree
Easy

Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:

    a binary tree in which the left and right subtrees of every node differ in height by no more than 1.

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: true

Example 2:

Input: root = [1,2,2,3,3,null,null,4,4]
Output: false

Example 3:

Input: root = []
Output: true
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def helper(root):
            if not root:
                return True, 0
              
            left = helper(root.left)
            
            if not left[0]:
                return False, 0
              
            right = helper(root.right)
            
            if not right[0]:
                return False, 0
              
            if abs(left[1]-right[1]) > 1:
                return False, 0
              
            return True, max(left[1], right[1])+1
			
        return helper(root)[0]
        
