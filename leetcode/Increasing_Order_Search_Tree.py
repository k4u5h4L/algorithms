'''
Increasing Order Search Tree
Easy

Given the root of a binary search tree, rearrange the tree in in-order so that the 
leftmost node in the tree is now the root of the tree, and every node has no left child and only 
one right child.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        if root == None:
            return root
        tree  =[]
        
        self.make_bst(root, tree)
        
        res = TreeNode(0)
        cur = res
        
        for node in tree:
            cur.right = TreeNode(node)
            cur = cur.right
        res = res.right
        return res
    
    def make_bst(self, root, tree):
        if root == None:
            return None
        self.make_bst(root.left, tree)
        tree.append(root.val)
        self.make_bst(root.right, tree)