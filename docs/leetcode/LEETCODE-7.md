
# leetcode programs:
## Page: 7
## Symmetric Tree

```py
'''
Symmetric Tree
Easy

Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

 

Example 1:

Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:

Input: root = [1,2,2,null,3,null,3]
Output: false
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root == None:
            return True
        
        return self.is_symmetric(root.left, root.right)
    
    def is_symmetric(self, left, right):
        if left == None or right == None:
            return left == right
        
        if left.val != right.val:
            return False
        
        return self.is_symmetric(left.left, right.right) and self.is_symmetric(left.right, right.left)

```

## Search in a Binary Search Tree

```py
'''
Search in a Binary Search Tree
Easy

You are given the root of a binary search tree (BST) and an integer val.

Find the node in the BST that the node's value equals val and return the subtree rooted with that node. If such a node does not exist, return null.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        return self.searchNode(root, val)
        
    def searchNode(self, root, val):
        if root == None:
            return None
        
        if val == root.val:
            return root
        elif val < root.val:
            return self.searchNode(root.left, val)
        else:
            return self.searchNode(root.right, val)
```

## Invert Binary Tree

```py
'''
Invert Binary Tree
Easy

Given the root of a binary tree, invert the tree, and return its root.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if root == None:
            return root
        
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        
        root.right = left
        root.left = right
        
        return root
```

## Is Subsequence

```py
'''
Is Subsequence
Easy

Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

 

Example 1:

Input: s = "abc", t = "ahbgdc"
Output: true

Example 2:

Input: s = "axc", t = "ahbgdc"
Output: false
'''


class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        for char in s:
            while i < len(t) and t[i] != char:
                i += 1
                
            if i >= len(t):
                return False
            elif t[i] == char:
                i += 1
                continue
                
        return True

```

## Find Minimum in Rotated Sorted Array

```py
'''
Find Minimum in Rotated Sorted Array
Medium

Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

    [4,5,6,7,0,1,2] if it was rotated 4 times.
    [0,1,2,4,5,6,7] if it was rotated 7 times.

Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
'''


class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        
        return nums[left]

```

## Climbing Stairs

```py
'''
Climbing Stairs
Easy

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
'''

import math
class Solution:
    def climbStairs(self, n: int) -> int:
        return self.total_ways(0, n, {})
        
    def total_ways(self, start, end, memo):
        if start in memo:
            return memo[start]
        if start == end:
            return 1
        elif start > end:
            return 0
        else:
            memo[start] = self.total_ways(start+1, end, memo) + self.total_ways(start+2, end, memo)
            return memo[start]
```

## Plus One

```py
'''
Plus One
Easy

Given a non-empty array of decimal digits representing a non-negative integer, increment one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contains a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.
'''

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        res = [str(num) for num in digits]
        res = str(int(''.join(res)) + 1)
        return [int(char) for char in res]
```

## Power of Two

```py
'''
Power of Two
Easy

Given an integer n, return true if it is a power of two. Otherwise, return false.

An integer n is a power of two, if there exists an integer x such that n == 2x.
'''

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1:
            return True
        while n > 1:
            if n % 2 != 0:
                return False
            n = int(n / 2)
        return True
```

## Maximum Product of Two Elements in an Array

```py
'''
Maximum Product of Two Elements in an Array
Easy
Given the array of integers nums, you will choose two different indices i and j of that array. Return the maximum value of (nums[i]-1)*(nums[j]-1). 
'''

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxEle = max(nums)
        nums.pop(nums.index(maxEle))
        secMax = max(nums)
        
        return (maxEle-1) * (secMax - 1)
```

## Valid Sudoku

```py
'''
Valid Sudoku
Medium

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

    Each row must contain the digits 1-9 without repetition.
    Each column must contain the digits 1-9 without repetition.
    Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Note:

    A Sudoku board (partially filled) could be valid but is not necessarily solvable.
    Only the filled cells need to be validated according to the mentioned rules.

'''

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        for i in range(0, 9):
            for j in range(0, 9):
                if not self.isValid(board, i, j):
                    return False
        return True
    
    def isValid(self, arr, row, col):
        return (self.notInRow(arr, row) and self.notInCol(arr, col) and
            self.notInBox(arr, row - row % 3, col - col % 3))
    
    def notInBox(self, arr, startRow, startCol):
        st = set()
        for row in range(0, 3):
            for col in range(0, 3):
                curr = arr[row + startRow][col + startCol]
                if curr in st:
                    return False
                if curr != '.':
                    st.add(curr)
        return True
    
    def notInCol(self, arr, col):
        st = set()
        for i in range(0, 9):
            if arr[i][col] in st:
                return False
            if arr[i][col] != '.':
                st.add(arr[i][col])

        return True
    
    def notInRow(self, arr, row):
        st = set()

        for i in range(0, 9):
            if arr[row][i] in st:
                return False
            if arr[row][i] != '.':
                st.add(arr[row][i])

        return True
```

## Subtract the Product and Sum of Digits of an Integer

```py
'''
Subtract the Product and Sum of Digits of an Integer
Easy
Given an integer number n, return the difference between the product of its digits and the sum of its digits. 
'''

class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        n = [int(char) for char in str(n)]
        print(n)
        
        pdt = 1
        
        for i in n:
            pdt *= i
            
        return (pdt - sum(n))
        
```

## Relative Sort Array

```py
'''
Relative Sort Array
Easy

Given two arrays arr1 and arr2, the elements of arr2 are distinct, and all elements in arr2 are also in arr1.

Sort the elements of arr1 such that the relative ordering of items in arr1 are the same as in arr2. Elements that do not appear in arr2 should be placed at the end of arr1 in ascending order.

 

Example 1:

Input: arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
Output: [2,2,2,1,4,3,3,9,6,7,19]

Example 2:

Input: arr1 = [28,6,22,8,44,17], arr2 = [22,28,8,6]
Output: [22,28,8,6,17,44]
'''


class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        res = []
        
        for n in arr2:
            while n in arr1:
                res.append(n)
                arr1.remove(n)
                
        if len(arr1) == 0:
            return res
        
        arr1.sort()
        res.extend(arr1)
        
        return res

```

## Merge Strings Alternately

```py
'''
Merge Strings Alternately
Easy

You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, starting with word1. If a string is longer than the other, append the additional letters onto the end of the merged string.

Return the merged string.

 

Example 1:

Input: word1 = "abc", word2 = "pqr"
Output: "apbqcr"
Explanation: The merged string will be merged as so:
word1:  a   b   c
word2:    p   q   r
merged: a p b q c r

Example 2:

Input: word1 = "ab", word2 = "pqrs"
Output: "apbqrs"
Explanation: Notice that as word2 is longer, "rs" is appended to the end.
word1:  a   b 
word2:    p   q   r   s
merged: a p b q   r   s

Example 3:

Input: word1 = "abcd", word2 = "pq"
Output: "apbqcd"
Explanation: Notice that as word1 is longer, "cd" is appended to the end.
word1:  a   b   c   d
word2:    p   q 
merged: a p b q c   d
'''


class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        top1 = 0
        top2 = 0
        turn = True
        res = ""
        
        while top1 < len(word1) and top2 < len(word2):
            if turn == True:
                res += word1[top1]
                top1 += 1
                turn = False
            else:
                res += word2[top2]
                top2 += 1
                turn = True
                
        while top1 < len(word1):
            res += word1[top1]
            top1 += 1
        
        while top2 < len(word2):
            res+= word2[top2]
            top2 += 1
            
        return res

```

## Search Insert Position

```py
'''
Search Insert Position
Easy

Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.
'''

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        pos = -1
        possible_place = -1
        while left <= right:
            mid = left + (right - left) // 2
            possible_place = mid
            
            if nums[mid] == target:
                pos = mid
                break
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        if nums[possible_place] < target:
            return possible_place + 1
        else:
            return possible_place

```

## Minimum Changes To Make Alternating Binary String

```py
'''
Minimum Changes To Make Alternating Binary String
Easy

You are given a string s consisting only of the characters '0' and '1'. In one operation, you can change any '0' to '1' or vice versa.

The string is called alternating if no two adjacent characters are equal. For example, the string "010" is alternating, while the string "0100" is not.

Return the minimum number of operations needed to make s alternating.

 

Example 1:

Input: s = "0100"
Output: 1
Explanation: If you change the last character to '1', s will be "0101", which is alternating.

Example 2:

Input: s = "10"
Output: 0
Explanation: s is already alternating.

Example 3:

Input: s = "1111"
Output: 2
Explanation: You need two operations to reach "0101" or "1010".
'''


class Solution:
    def minOperations(self, s: str) -> int:
        dp0 = int(s[0])
        dp1 = 1 - dp0
        
        for i in range(1, len(s)):
            if s[i] == '1':
                dp0, dp1 = 1 + dp1, dp0
            else:
                dp1, dp0 = 1 + dp0, dp1
                
        return min(dp0, dp1)

```

## Implement Stack using Queues

```py
'''
Implement Stack using Queues
Easy

Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

Implement the MyStack class:

    void push(int x) Pushes element x to the top of the stack.
    int pop() Removes the element on the top of the stack and returns it.
    int top() Returns the element on the top of the stack.
    boolean empty() Returns true if the stack is empty, false otherwise.

Notes:

    You must use only standard operations of a queue, which means that only push to back, peek/pop from front, size and is empty operations are valid.
    Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.

 

Example 1:

Input
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 2, 2, false]

Explanation
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False

 

Constraints:

    1 <= x <= 9
    At most 100 calls will be made to push, pop, top, and empty.
    All the calls to pop and top are valid.
'''


class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q1 = []
        self.q2 = []
        

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.q2 = self.q1.copy()
        self.q1 = []
        self.q1.append(x)
        self.q1.extend(self.q2)
        

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.q1.pop(0)
        

    def top(self) -> int:
        """
        Get the top element.
        """
        return self.q1[0]
        

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return len(self.q1) == 0
        


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

```

## Maximum Product Subarray

```py
'''
Maximum Product Subarray
Medium

Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

It is guaranteed that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.
'''

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_pdt = max(nums)
        cur_min = 1
        cur_max = 1
        
        for num in nums:
            if num == 0:
                cur_min = 1
                cur_max = 1
            else:
                temp = cur_max * num
                cur_max = max(num * cur_max, num * cur_min, num)
                cur_min = min(temp, num * cur_min, num)
                max_pdt = max(max_pdt, cur_max)
        return max_pdt
```

## Largest Substring Between Two Equal Characters

```py
'''
Largest Substring Between Two Equal Characters
Easy

Given a string s, return the length of the longest substring between two equal characters, excluding the two characters. If there is no such substring return -1.

A substring is a contiguous sequence of characters within a string.

 

Example 1:

Input: s = "aa"
Output: 0
Explanation: The optimal substring here is an empty substring between the two 'a's.

Example 2:

Input: s = "abca"
Output: 2
Explanation: The optimal substring here is "bc".

Example 3:

Input: s = "cbzxy"
Output: -1
Explanation: There are no characters that appear twice in s.

Example 4:

Input: s = "cabbac"
Output: 4
Explanation: The optimal substring here is "abba". Other non-optimal substrings include "bb" and "".
'''


class Solution:
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        res = -1
        
        if len(s) == 1:
            return res
        
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    res = max(res, len(s[i+1:j]))
                    
        return res

```

## Transpose Matrix

```py
'''
Transpose Matrix
Easy

Given a 2D integer array matrix, return the transpose of matrix.

The transpose of a matrix is the matrix flipped over its main diagonal, switching the matrix's row and column indices.
'''

class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
```

## Max Consecutive Ones

```py
'''
Max Consecutive Ones
Easy

Given a binary array nums, return the maximum number of consecutive 1's in the array.

 

Example 1:

Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.

Example 2:

Input: nums = [1,0,1,1,0,1]
Output: 2
'''


class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
                
        elif 1 not in nums:
            return 0
        
        max_len = 1
        cur_len = 0
        
        for val in nums:
            if val == 1:
                cur_len += 1
            else:
                cur_len = 0
                
            max_len = max(max_len, cur_len)
            
        return max_len

```

## Reverse Words in a String

```py
'''
Reverse Words in a String
Medium

Given an input string s, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that s may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

 

Example 1:

Input: s = "the sky is blue"
Output: "blue is sky the"

Example 2:

Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.

Example 3:

Input: s = "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.

Example 4:

Input: s = "  Bob    Loves  Alice   "
Output: "Alice Loves Bob"

Example 5:

Input: s = "Alice does not even like bob"
Output: "bob like even not does Alice"
'''


class Solution:
    def reverseWords(self, s: str) -> str:
        s = re.sub("\s+", " ", s.strip())
        
        words = s.split(" ")[::-1]
        
        return " ".join(words)

```

## Number of 1 Bits

```py
'''
Number of 1 Bits
Easy

Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Note:

    Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
    In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.

'''

class Solution:
    def hammingWeight(self, n: int) -> int:
        res = []
        
        while n > 0:
            res.append(n % 2)
            n = int(n/2)
        return res.count(1)
```

## Find the Difference

```py
'''
Find the Difference
Easy

You are given two strings s and t.

String t is generated by random shuffling string s and then add one more letter at a random position.

Return the letter that was added to t.

 

Example 1:

Input: s = "abcd", t = "abcde"
Output: "e"
Explanation: 'e' is the letter that was added.

Example 2:

Input: s = "", t = "y"
Output: "y"

Example 3:

Input: s = "a", t = "aa"
Output: "a"

Example 4:

Input: s = "ae", t = "aea"
Output: "a"
'''


class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        s = list(s)
        s.sort()
        
        t = list(t)
        t.sort()
        
        for a1, a2 in zip(s, t):
            if a1 != a2:
                return a2
            
        return t[-1]

```

## Binary Number with Alternating Bits

```py
'''
Binary Number with Alternating Bits
Easy

Given a positive integer, check whether it has alternating bits: namely, if two adjacent bits will always have different values.

 

Example 1:

Input: n = 5
Output: true
Explanation: The binary representation of 5 is: 101

Example 2:

Input: n = 7
Output: false
Explanation: The binary representation of 7 is: 111.

Example 3:

Input: n = 11
Output: false
Explanation: The binary representation of 11 is: 1011.

Example 4:

Input: n = 10
Output: true
Explanation: The binary representation of 10 is: 1010.

Example 5:

Input: n = 3
Output: false
'''


class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        binary = bin(n)[2:]
        
        if len(binary) == 1:
            return True
        
        for i in range(len(binary) - 1):
            if binary[i] == binary[i+1]:
                return False
            
        return True

```

## Path Sum II

```py
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

```

## Isomorphic Strings

```py
'''
Isomorphic Strings
Easy

Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

 

Example 1:

Input: s = "egg", t = "add"
Output: true

Example 2:

Input: s = "foo", t = "bar"
Output: false

Example 3:

Input: s = "paper", t = "title"
Output: true
'''

class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        memo = {}
        
        if len(s) != len(t):
            return False
        
        for a, b in zip(s, t):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
        
        memo = {}
        
        for a, b in zip(t, s):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
        
        return True

```

## Insert into a Binary Search Tree

```py
'''
Insert into a Binary Search Tree
Medium

You are given the root node of a binary search tree (BST) and a value to insert into the tree. Return the root node of the BST after the insertion. It is guaranteed that the new value does not exist in the original BST.

Notice that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after insertion. You can return any of them.

 

Example 1:

Input: root = [4,2,7,1,3], val = 5
Output: [4,2,7,1,3,5]
Explanation: Another accepted tree is:

Example 2:

Input: root = [40,20,60,10,30,50,70], val = 25
Output: [40,20,60,10,30,50,70,null,null,25]

Example 3:

Input: root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
Output: [4,2,7,1,3,5]
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if root == None:
            return TreeNode(val)
        
        cur = root
        prev = cur
        
        while cur != None:
            prev = cur
            if val > cur.val:
                cur = cur.right
            elif val < cur.val:
                cur = cur.left
                
        if val > prev.val:
            prev.right = TreeNode(val)
        else:
            prev.left = TreeNode(val)
            
        return root

```

## Counting Bits

```py
'''
Counting Bits
Easy

Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
'''

class Solution:
    def countBits(self, n: int) -> List[int]:
        res = []
        for i in range(n+1):
            bits = self.getBin(i)
            res.append(bits.count(1))
        return res
    
    def getBin(self, decimal):
        result = []
        while decimal > 0:
            if decimal % 2 == 0:
                result.append(0)
            else:
                result.append(1)
            decimal = int(decimal / 2)

        result.reverse()
        return result
```

## Min Stack

```py
'''
Min Stack
Easy

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

    MinStack() initializes the stack object.
    void push(val) pushes the element val onto the stack.
    void pop() removes the element on the top of the stack.
    int top() gets the top element of the stack.
    int getMin() retrieves the minimum element in the stack.

 

Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
'''

class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        

    def push(self, val: int) -> None:
        self.stack.append(val)
        

    def pop(self) -> None:
        self.stack.pop()
        

    def top(self) -> int:
        return self.stack[-1]
        

    def getMin(self) -> int:
        return min(self.stack)
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

```

## Intersection of Two Linked Lists

```py
'''
Intersection of Two Linked Lists
Easy

Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:

a1 -> a2 -------↓
                -> c1 -> c2 -> c3
b1 -> b2 -> b3 -↑

It is guaranteed that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        cur_a = headA
        cur_b = headB
        
        while cur_a != cur_b:
            if cur_a == None:
                cur_a = headB
            else:
                cur_a = cur_a.next
                
            if cur_b == None:
                cur_b = headA
            else:
                cur_b = cur_b.next
        
        return cur_a

```

