'''
Find Mode in Binary Search Tree
Easy

Given the root of a binary search tree (BST) with duplicates, return all the mode(s) (i.e., the most frequently occurred element) in it.

If the tree has more than one mode, return them in any order.

Assume a BST is defined as follows:

    The left subtree of a node contains only nodes with keys less than or equal to the node's key.
    The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
    Both the left and right subtrees must also be binary search trees.

 

Example 1:

Input: root = [1,null,2,2]
Output: [2]

Example 2:

Input: root = [0]
Output: [0]
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:        
        if root == None:
            return []
        
        elif root.left == None and root.right == None:
            return [root.val]
        
        vals = []
        
        self.inorder(root, vals)
        
        dic = {}
        
        for val in vals:
            if val in dic:
                dic[val] += 1
            else:
                dic[val] = 1
        
        res = []
        cur_max = 0
                
        for key, value in dic.items():
            if cur_max < value:
                res = []
                res.append(key)
                cur_max = value
            elif cur_max == value:
                res.append(key)
                
        return res
    
    def inorder(self, root, vals):
        if root == None:
            return
        
        self.inorder(root.left, vals)
        vals.append(root.val)
        self.inorder(root.right, vals)
