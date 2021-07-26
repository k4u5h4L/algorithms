'''
Path Sum II
Medium

Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where each path's sum equals targetSum.

A leaf is a node with no children.

 

Example 1:

Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]

Example 2:

Input: root = [1,2,3], targetSum = 5
Output: []

Example 3:

Input: root = [1,2], targetSum = 0
Output: []
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        if root == None:
            return []
        
        paths = []
        
        self.traverse(root, 0, targetSum, paths, "")
        
        return paths
    
    def traverse(self, root, cur_sum, targetSum, paths, path):
        if root == None:
            return
        
        if root.left == None and root.right == None and cur_sum + root.val == targetSum:
            path += f"->{root.val}"
            p = [int(char) for char in path.split("->") if char != ""]
            paths.append(p)
            return
            
        if path == "":
            self.traverse(root.left, cur_sum + root.val, targetSum, paths, f"{root.val}")
            self.traverse(root.right, cur_sum + root.val, targetSum, paths, f"{root.val}")
        else:
            self.traverse(root.left, cur_sum + root.val, targetSum, paths, f"{path}->{root.val}")
            self.traverse(root.right, cur_sum + root.val, targetSum, paths, f"{path}->{root.val}")
