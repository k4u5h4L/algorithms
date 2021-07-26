'''
Binary Tree Paths
Easy

Given the root of a binary tree, return all root-to-leaf paths in any order.

A leaf is a node with no children.

 

Example 1:

Input: root = [1,2,3,null,5]
Output: ["1->2->5","1->3"]

Example 2:

Input: root = [1]
Output: ["1"]
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if root == None:
            return ['']
        if root.right == None and root.left == None:
            return f"{root.val}"
        
        paths = []
        
        self.inorder(root, "", paths)
        
        return paths
    
    def inorder(self, root, path, paths):
        if root == None:
            return
        elif root.left == None and root.right == None:
            path += f"->{root.val}"
            paths.append(path)
            return
        if path == "":
            self.inorder(root.left, f"{root.val}", paths)
            self.inorder(root.right, f"{root.val}", paths)
        else:
            self.inorder(root.left, f"{path}->{root.val}", paths)
            self.inorder(root.right, f"{path}->{root.val}", paths)
