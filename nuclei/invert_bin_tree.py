# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def invertTree(root):
    if root == None:
        return root
        
    left = invertTree(root.left)
    right = invertTree(root.right)
        
    root.right = left
    root.left = right
        
    return root

def main():
    root = TreeNode(5)
    root.left = TreeNode(4)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(4)
    root.right = TreeNode(6)
    root.right.right = TreeNode(7)
    root.right.left = TreeNode(4)

    res = invertTree(root)

    print(root.left.val)
    print(root.right.val)

main()