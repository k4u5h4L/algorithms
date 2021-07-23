'''
Kth Smallest Element in a BST
Medium

Given the root of a binary search tree, and an integer k, return the kth (1-indexed) smallest element in the tree.

 

Example 1:

Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        count = []
        
        self.inorder(root, k, count)
        
        return count[k-1]
    
    def inorder(self, root, k, count):
        if root == None or len(count) == k:
            return
        
        self.inorder(root.left, k, count)
        count.append(root.val)
        self.inorder(root.right, k, count)
        
        
