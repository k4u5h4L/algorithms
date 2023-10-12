
# leetcode programs:
## Page: 6
## Verifying an Alien Dictionary

```py
'''
Verifying an Alien Dictionary
Easy

In an alien language, surprisingly they also use english lowercase letters, but possibly in a different order. 
The order of the alphabet is some permutation of lowercase letters.

Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only 
if the given words are sorted lexicographicaly in this alien language.

 

Example 1:

Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.

Example 2:

Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
Output: false
Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], hence the sequence is unsorted.

Example 3:

Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
Output: false
Explanation: The first three characters "app" match, and the second string is shorter (in size.) According to lexicographical 
rules "apple" > "app", because 'l' > '∅', where '∅' is defined as the blank character which is less than any other character
'''

class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        dic = {}
        
        for i, a in enumerate(order):
            dic[a] = i
            
        for i in range(len(words)-1):
            for j in range(len(words[i])):
                if j >= len(words[i+1]):
                    return False
                
                if words[i][j] != words[i+1][j]:
                    if dic[words[i][j]] > dic[words[i+1][j]]:
                        return False
                    
                    break
        
        return True

```

## Check if Word Equals Summation of Two Words

```py
'''
Check if Word Equals Summation of Two Words
Easy

The letter value of a letter is its position in the alphabet starting from 0 (i.e. 'a' -> 0, 'b' -> 1, 'c' -> 2, etc.).

The numerical value of some string of lowercase English letters s is the concatenation of the letter values of each letter in s, which is then converted into an integer.

    For example, if s = "acb", we concatenate each letter's letter value, resulting in "021". After converting it, we get 21.

You are given three strings firstWord, secondWord, and targetWord, each consisting of lowercase English letters 'a' through 'j' inclusive.

Return true if the summation of the numerical values of firstWord and secondWord equals the numerical value of targetWord, or false otherwise.
'''

class Solution:
    def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:
        word = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
        s1 = [str(word[char])for char in firstWord]
        s1 = int(''.join(s1))
        s2 = [str(word[char])for char in secondWord]
        s2 = int(''.join(s2))
        s = [str(word[char])for char in targetWord]
        s = int(''.join(s))
        return (s1 + s2) == s
```

## Find the Duplicate Number

```py
'''
Find the Duplicate Number
Medium

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.
'''

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            if nums[abs(nums[i]) - 1] < 0:
                return abs(nums[i])
            else:
                nums[abs(nums[i]) - 1] = -nums[abs(nums[i]) - 1]
                
        return -1
#         nums.sort()
        
#         for i in range(1, len(nums)):
#             if nums[i-1] == nums[i]:
#                 return nums[i]
```

## Balanced Binary Tree

```py
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
        

```

## Odd Even Linked List

```py
'''
Odd Even Linked List
Medium

Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

The first node is considered odd, and the second node is even, and so on.

Note that the relative order inside both the even and odd groups should remain as it was in the input.

You must solve the problem in O(1) extra space complexity and O(n) time complexity.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        odd = head
        even = head.next
        cur = head
        
        while cur != None and cur.next != None:
            lnext = cur.next
            cur.next = cur.next.next
            cur = lnext
        while odd.next != None:
            odd = odd.next
        odd.next = even
        return head
```

## XOR Operation in an Array

```py
'''
XOR Operation in an Array
Easy

Given an integer n and an integer start.

Define an array nums where nums[i] = start + 2*i (0-indexed) and n == nums.length.

Return the bitwise XOR of all elements of nums.
'''

class Solution:
    def xorOperation(self, n: int, start: int) -> int:
        exor = 0
        for i in range(n):
            exor = exor ^ (start + 2*i)
            
        return exor
```

## Check If N and Its Double Exist

```py
'''
Check If N and Its Double Exist
Easy

Given an array arr of integers, check if there exists two integers N and M such that N is the double of M ( i.e. N = 2 * M).

More formally check if there exists two indices i and j such that :

    i != j
    0 <= i, j < arr.length
    arr[i] == 2 * arr[j]

 

Example 1:

Input: arr = [10,2,5,3]
Output: true
Explanation: N = 10 is the double of M = 5,that is, 10 = 2 * 5.

Example 2:

Input: arr = [7,1,14,11]
Output: true
Explanation: N = 14 is the double of M = 7,that is, 14 = 2 * 7.

Example 3:

Input: arr = [3,1,7,11]
Output: false
Explanation: In this case does not exist N and M, such that N = 2 * M.
'''

class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        dic = {}
        
        for i, num in enumerate(arr):
            dic[num] = i
            
        for i, num in enumerate(arr):
            if (num * 2) in dic and i != dic[(num * 2)]:
                return True
            
        return False

```

## Count Negative Numbers in a Sorted Matrix

```py
'''
Count Negative Numbers in a Sorted Matrix
Easy

Given a m x n matrix grid which is sorted in non-increasing order both row-wise and column-wise, return the number of negative numbers in grid.
'''

class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] < 0:
                    count += 1
                # else:
                #     break
        return count
```

## Battleships in a Board

```py
'''
Battleships in a Board
Medium

Given an m x n matrix board where each cell is a battleship 'X' or empty '.', return the number of the battleships on board.

Battleships can only be placed horizontally or vertically on board. In other words, they can only be made of the shape 1 x k (1 row, k columns) or k x 1 (k rows, 1 column), where k can be of any size. At least one horizontal or vertical cell separates between two battleships (i.e., there are no adjacent battleships).

 

Example 1:

Input: board = [["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]
Output: 2

Example 2:

Input: board = [["."]]
Output: 0
'''


# Naive approach

class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        count = 0
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 'X':
                    count += self.dfs(board, i, j)
        
        return count
    
    def dfs(self, board, i, j):
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[i]) or board[i][j] != 'X':
            return 0
        
        board[i][j] = '.'
        
        self.dfs(board, i+1, j)
        self.dfs(board, i-1, j)
        self.dfs(board, i, j+1)
        self.dfs(board, i, j-1)
        
        return 1
      
      

      
# optimised approach


class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        count = 0
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == '.':
                    continue
                if i > 0 and board[i-1][j] == 'X':
                    continue
                if j > 0 and board[i][j-1] == 'X':
                    continue
                
                count += 1
        
        return count

```

## The kth Factor of n

```py
'''
The kth Factor of n
Medium

Given two positive integers n and k.

A factor of an integer n is defined as an integer i where n % i == 0.

Consider a list of all factors of n sorted in ascending order, return the kth factor in this list or return -1 if n has less than k factors.
'''

class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        factors = []
        for i in range(1, n+1):
            if n % i == 0:
                factors.append(i)    
                
        if (len(factors) < k):
            return -1
        
        return factors[k - 1]
```

## Valid Palindrome II

```py
'''
Valid Palindrome II
Easy

Given a string s, return true if the s can be palindrome after deleting at most one character from it.
'''

class Solution:
    def validPalindrome(self, s: str) -> bool:
        i = 0
        j = len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return self.isPalindrome(s, i+1, j) or self.isPalindrome(s, i, j-1)
            i += 1
            j -= 1
        return True
    
    def isPalindrome(self, s, i, j):
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

```

## Check If a Word Occurs As a Prefix of Any Word in a Sentence

```js
/*
Check If a Word Occurs As a Prefix of Any Word in a Sentence
Easy

Given a sentence that consists of some words separated by a single space, and a searchWord.

You have to check if searchWord is a prefix of any word in sentence.

Return the index of the word in sentence where searchWord is a prefix of this word (1-indexed).

If searchWord is a prefix of more than one word, return the index of the first word (minimum index). If there is no such word return -1.

A prefix of a string S is any leading contiguous substring of S.
*/

/**
 * @param {string} sentence
 * @param {string} searchWord
 * @return {number}
 */
var isPrefixOfWord = function (sentence, searchWord) {
    const words = sentence.split(" ");
    const regex = new RegExp(`^(${searchWord})`, "i");

    for (let i = 0; i < words.length; i++) {
        if (regex.test(words[i])) {
            return i + 1;
        }
    }

    return -1;
};

```

## Number Complement

```py
'''
Number Complement
Easy

The complement of an integer is the integer you get when you flip all the 0's to 1's and all the 1's to 0's in its binary representation.

    For example, The integer 5 is "101" in binary and its complement is "010" which is the integer 2.

Given an integer num, return its complement.

 

Example 1:

Input: num = 5
Output: 2
Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.

Example 2:

Input: num = 1
Output: 0
Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0. So you need to output 0.
'''


class Solution:
    def findComplement(self, num: int) -> int:
        binary = bin(num)[2:]
        
        b = ""
        
        for bit in binary:
            if bit == '1':
                b += '0'
            else:
                b += '1'
                
        dec = 0
        
        for i, char in enumerate(reversed(b)):
            if char == '1':
                dec += (2 ** i)
            
        return dec

```

## Maximum Depth of Binary Tree

```py
'''
Maximum Depth of Binary Tree
Easy

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

## Average Salary Excluding the Minimum and Maximum Salary

```py
'''
Average Salary Excluding the Minimum and Maximum Salary
Easy

Given an array of unique integers salary where salary[i] is the salary of the employee i.

Return the average salary of employees excluding the minimum and maximum salary.
'''

class Solution:
    def average(self, salary: List[int]) -> float:
        salary.pop(salary.index(min(salary)))
        salary.pop(salary.index(max(salary)))
        
        return sum(salary) / len(salary)
```

## Merge Two Sorted Lists

```py
'''
Merge Two Sorted Lists
Easy

Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        res = None

        while l1 != None and l2 != None:
            if l1.val < l2.val:
                if res == None:
                    res = ListNode(l1.val)
                else:
                    cur = res
                    while cur != None and cur.next != None:
                        cur = cur.next

                    cur.next = ListNode(val=l1.val)
                l1 = l1.next
            else:
                if res == None:
                    res = ListNode(l2.val)
                else:
                    cur = res
                    while cur != None and cur.next != None:
                        cur = cur.next

                    cur.next = ListNode(val=l2.val)
                l2 = l2.next
        
        while l1 != None:
            if res == None:
                res = ListNode(l1.val)
                l1 = l1.next
                continue
            cur = res
            while cur != None and cur.next != None:
                cur = cur.next
                    
            cur.next = ListNode(val=l1.val)
            l1 = l1.next
        while l2 != None:
            if res == None:
                res = ListNode(l2.val)
                l2 = l2.next
                continue
            cur = res
            while cur != None and cur.next != None:
                cur = cur.next
                    
            cur.next = ListNode(val=l2.val)
            l2 = l2.next
            
        return res
```

## Add to Array-Form of Integer

```py
'''
Add to Array-Form of Integer
Easy

The array-form of an integer num is an array representing its digits in left to right order.

    For example, for num = 1321, the array form is [1,3,2,1].

Given num, the array-form of an integer, and an integer k, return the array-form of the integer num + k.
'''

class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        res = [str(n) for n in num]
        res = str(int(''.join(res)) + k)
        return [int(char) for char in res]
```

## two sum

```py
'''
Two Sum
Easy

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.
'''

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [i, dic[nums[i]]]
            else:
                dic[target - nums[i]] = i
        return [-1, -1]
```

## Convert Binary Number in a Linked List to Integer

```py
'''
Convert Binary Number in a Linked List to Integer
Easy

Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        binary = []
        while head != None:
            binary.append(head.val)
            head = head.next
            
        deci = 0
        reversedBin = [bit for bit in reversed(binary)]
        
        for i in range(len(reversedBin)):
            if reversedBin[i] == 1:
                deci += (2 ** i)
                
        return deci
```

## Binary Tree Postorder Traversal

```py
'''
Binary Tree Postorder Traversal
Easy

Given the root of a binary tree, return the postorder traversal of its nodes' values.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        
        self.postorder(root, res)
        
        return res
    
    def postorder(self, root, res):
        if root == None:
            return
        
        self.postorder(root.left, res)
        self.postorder(root.right, res)
        res.append(root.val)
```

## Merge Two Binary Trees

```py
'''
Merge Two Binary Trees
Easy

You are given two binary trees root1 and root2.

Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.

Return the merged tree.

Note: The merging process must start from the root nodes of both trees.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if not root1:
            return root2
        if not root2:
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1
```

## Move Zeroes

```py
'''
Move Zeroes
Easy

Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.
'''

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        num_zeros = nums.count(0)
        for _ in range(num_zeros):
            nums.remove(0)
        for _ in range(num_zeros):
            nums.append(0)
```

## Remove Duplicates from Sorted List

```py
'''
Remove Duplicates from Sorted List
Easy

Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        cur = head
        while cur != None and cur.next != None:
            while cur.next != None and cur.val == cur.next.val:
                cur.next = cur.next.next
            cur = cur.next
        return head
```

## Diameter of Binary Tree

```py
'''
Diameter of Binary Tree
Easy

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.

 

Example 1:

Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3is the length of the path [4,2,1,3] or [5,2,1,3].

Example 2:

Input: root = [1,2]
Output: 1
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        if root == None:
            return 0
        
        res = [0]
        self.diameter(root, res)
        return res[0] - 1
    
    def diameter(self, root, res):
        if root == None:
            return 0

        left = self.diameter(root.left, res);
        right = self.diameter(root.right, res);
        res[0] = max(res[0], 1 + left + right);
        
        return max(left, right) + 1;

```

## Palindrome Linked List

```py
'''
Palindrome Linked List
Easy

Given the head of a singly linked list, return true if it is a palindrome.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        slow = head
        fast = head
        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
        slow = self.reverse_list(slow)
        fast = head
        while slow != None:
            if slow.val != fast.val:
                return False
            slow = slow.next
            fast = fast.next
        return True
            
    def reverse_list(self, head):
        prev = None
        while head != None:
            lnext = head.next
            head.next = prev
            prev = head
            head = lnext
        return prev
```

## Running Sum of 1d Array

```py
'''
Running Sum of 1d Array
Easy

Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).

Return the running sum of nums.
'''

class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        res = []
        for i in range(len(nums)):
            res.append(sum(nums[:i+1]))
            
        return res
```

## Element Appearing More Than 25% In Sorted Array

```py
'''
Element Appearing More Than 25% In Sorted Array
Easy

Given an integer array sorted in non-decreasing order, there is exactly one integer in the array that occurs 
more than 25% of the time, return that integer.
'''

class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        count = 0
        
        if len(arr) < 2:
            return arr[0]
        
        for i in arr:
            if arr.count(i) / len(arr) > 0.25:
                return i
            
        return 0
```

## Merge Sorted Array

```py
'''
 Merge Sorted Array
Easy

You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of 
elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. 
To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 
and should be ignored. nums2 has a length of n.
'''

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums = nums1.copy()
        l1 = 0
        l2 = 0
        top = 0
        while l1 < m and l2 < n:
            if nums[l1] < nums2[l2]:
                nums1[top] = nums[l1]
                l1 += 1
            else:
                nums1[top] = nums2[l2]
                l2 += 1
            top += 1
        
        while l1 < m:
            nums1[top] = nums[l1]
            l1 += 1
            top += 1
        
        while l2 < n:
            nums1[top] = nums2[l2]
            l2 += 1
            top += 1

```

## Concatenation of Array

```py
'''
Concatenation of Array
Easy

Given an integer array nums of length n, you want to create an array ans of length 2n where ans[i] == nums[i] and ans[i + n] == nums[i] for 0 <= i < n (0-indexed).

Specifically, ans is the concatenation of two nums arrays.

Return the array ans.

 

Example 1:

Input: nums = [1,2,1]
Output: [1,2,1,1,2,1]
Explanation: The array ans is formed as follows:
- ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]]
- ans = [1,2,1,1,2,1]

Example 2:

Input: nums = [1,3,2,1]
Output: [1,3,2,1,1,3,2,1]
Explanation: The array ans is formed as follows:
- ans = [nums[0],nums[1],nums[2],nums[3],nums[0],nums[1],nums[2],nums[3]]
- ans = [1,3,2,1,1,3,2,1]
'''


class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        return nums + nums

```

## Ugly Number

```py
'''
Ugly Number
Easy

An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

Given an integer n, return true if n is an ugly number.

 

Example 1:

Input: n = 6
Output: true
Explanation: 6 = 2 × 3

Example 2:

Input: n = 8
Output: true
Explanation: 8 = 2 × 2 × 2

Example 3:

Input: n = 14
Output: false
Explanation: 14 is not ugly since it includes the prime factor 7.

Example 4:

Input: n = 1
Output: true
Explanation: 1 has no prime factors, therefore all of its prime factors are limited to 2, 3, and 5.
'''

class Solution:
    def isUgly(self, n: int) -> bool:
        if n == 0:
            return False
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            elif n % 3 == 0:
                n = n // 3
            elif n % 5 == 0:
                n = n // 5
            else:
                return False
        
        return True

```

