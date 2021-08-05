'''
Subtree of Another Tree
Easy

Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

 

Example 1:

Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true

Example 2:

Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        if root == None and subRoot != None or root != None and subRoot == None:
            return False
        
        res = [False]
        
        self.get_subtree(root, subRoot, res)
        
        return res[0]
    
    # keep traversing the root until the value of a node equals the root node val of the subtree
    def get_subtree(self, root, subRoot, res):
        if root == None:
            return
        
        if root.val == subRoot.val:
          # if equal, then check for the rest of the tree
            if self.is_same_tree(root, subRoot) == True:
                res[0] = True
                return
            
        self.get_subtree(root.left, subRoot, res)
        self.get_subtree(root.right, subRoot, res)
        
    # if the rest of the tree is also same, then return true else false
    def is_same_tree(self, root1, root2):
        if root1 == None and root2 != None or root1 != None and root2 == None:
            return False
        
        elif root1 == None and root2 == None:
            return True
        
        if root1.val == root2.val:
            return self.is_same_tree(root1.left, root2.left) and self.is_same_tree(root1.right, root2.right)
        else:
            return False
