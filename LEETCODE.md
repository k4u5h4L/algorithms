
# leetcode programs:

## Binary Tree Preorder Traversal

```py
'''
 Binary Tree Preorder Traversal
Easy

Given the root of a binary tree, return the preorder traversal of its nodes' values.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        
        self.preorder(root, res)
        
        return res
    
    def preorder(self, root, res):
        if root == None:
            return
        
        res.append(root.val)
        self.preorder(root.left, res)
        self.preorder(root.right, res)
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

## Container With Most Water

```py
'''
Container With Most Water
Medium

Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

Notice that you may not slant the container.
'''

class Solution:
    def maxArea(self, height: List[int]) -> int:
        left_ptr = 0
        max_area = 0
        right_ptr = len(height)-1
        
        while left_ptr < right_ptr:
            if height[left_ptr] < height[right_ptr]:
                max_area = max(max_area, height[left_ptr] * (right_ptr - left_ptr))
                left_ptr += 1
            else:
                max_area = max(max_area, height[right_ptr] * (right_ptr - left_ptr))
                right_ptr -= 1
        
        return max_area
```

## Word Pattern

```py
'''
Word Pattern
Easy

Given a pattern and a string s, find if s follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s.

 

Example 1:

Input: pattern = "abba", s = "dog cat cat dog"
Output: true

Example 2:

Input: pattern = "abba", s = "dog cat cat fish"
Output: false

Example 3:

Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false

Example 4:

Input: pattern = "abba", s = "dog dog dog dog"
Output: false
'''

class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        pattern = [char for char in pattern]
        s = s.split(" ")
        
        if len(pattern) != len(s):
            return False
        
        memo = {}
        
        for a, b in zip(pattern, s):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
                
        memo = {}
        
        for a, b in zip(s, pattern):
            if a in memo:
                if memo[a] != b:
                    return False
            else:
                memo[a] = b
                
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

## Largest Odd Number in String

```py
'''
Largest Odd Number in String
Easy

You are given a string num, representing a large integer. Return the largest-valued odd integer (as a string) that is a non-empty substring of num, or an empty string "" if no odd integer exists.

A substring is a contiguous sequence of characters within a string.

 

Example 1:

Input: num = "52"
Output: "5"
Explanation: The only non-empty substrings are "5", "2", and "52". "5" is the only odd number.

Example 2:

Input: num = "4206"
Output: ""
Explanation: There are no odd numbers in "4206".

Example 3:

Input: num = "35427"
Output: "35427"
Explanation: "35427" is already an odd number.
'''

class Solution:
    def largestOddNumber(self, num: str) -> str:
        right = len(num)
        
        while right >= 0:
            if int(num[right - 1]) % 2 != 0:
                return num[:right]
            right -= 1
        
        return ""

```

## Product of Array Except Self

```py
'''
Product of Array Except Self
Medium

Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.
'''

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        left_pdt = [1]
        
        right_pdt = nums.copy()
        right_pdt[len(nums)-1] = 1

        for i in range(len(nums)-1):
            left_pdt.append(nums[i] * left_pdt[i])
        i = len(nums) - 2

        while i >= 0:
            right_pdt[i] = nums[i+1] * right_pdt[i+1]
            i -= 1

        res = []

        for i in range(len(nums)):
            res.append(left_pdt[i] * right_pdt[i])

        return res
```

## Rotate Array

```py
'''
Rotate Array
Medium

Given an array, rotate the array to the right by k steps, where k is non-negative.
'''

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for _ in range(k):
            temp = nums.pop()
            nums.insert(0, temp)
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

## Max Area of Island

```py
'''
Max Area of Island
Medium

You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

 

Example 1:

Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.

Example 2:

Input: grid = [[0,0,0,0,0,0,0,0]]
Output: 0
'''


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        max_area = 0
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    max_area = max(max_area, self.dfs(grid, i, j, [0]))
        
        return max_area
    
    def dfs(self, grid, i, j, area):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] != 1:
            return 0
        grid[i][j] = 0
        
        area[0] += 1
        
        self.dfs(grid, i+1, j, area)
        self.dfs(grid, i-1, j, area)
        self.dfs(grid, i, j+1, area)
        self.dfs(grid, i, j-1, area)
        
        return area[0]

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

## Valid Palindrome

```js
/*
Valid Palindrome
Easy

Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
*/

/**
 * @param {string} s
 * @return {boolean}
 */
var isPalindrome = function (s) {
    s = s.toLowerCase();
    s = s.replace(/[\W_]/gim, "");
    rev_s = s.split("").reverse().join("");
    return s == rev_s;
};

```

## Missing Number

```py
'''
Missing Number
Easy

Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?
'''

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        expected_sum = len(nums)*(len(nums)+1)//2
        actual_sum = sum(nums)
        return expected_sum - actual_sum
```

## Contains Duplicate

```py
'''
Contains Duplicate
Easy

Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
'''

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        dups = {}
        for num in nums:
            if num in dups:
                return True
            else:
                dups[num] = True
        return False
```

## Backspace String Compare

```py
'''
Backspace String Compare
Easy

Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.

Note that after backspacing an empty text, the text will continue empty.
'''

class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        s = [char for char in s]
        t = [char for char in t]
        stack_s = []
        stack_t = []
        
        for i in s:
            if i == '#':
                if len(stack_s) == 0:
                    continue
                stack_s.pop()
            else:
                stack_s.append(i)
        for i in t:
            if i == '#':
                if len(stack_t) == 0:
                    continue
                stack_t.pop()
            else:
                stack_t.append(i)
                
        return stack_s == stack_t
```

## Add Digits

```py
'''
Add Digits
Easy

Given an integer num, repeatedly add all its digits until the result has only one digit, and return it.
'''

class Solution:
    def addDigits(self, num: int) -> int:
        sum1 = self.getSum(num)
        while sum1 > 9:
            sum1 = self.getSum(sum1)
            
        return sum1
    
    def getSum(self, num):
        sum1 = 0
        while num > 0:
            sum1 += num % 10
            num = int(num / 10)
            
        return sum1
```

## Two Sum II- Input array is sorted

```py
'''
Two Sum II - Input array is sorted
Easy

Given an array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number.

Return the indices of the two numbers (1-indexed) as an integer array answer of size 2, where 1 <= answer[0] < answer[1] <= numbers.length.

The tests are generated such that there is exactly one solution. You may not use the same element twice.
'''

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left <= right:
            cur_sum = numbers[left] + numbers[right]
            if cur_sum < target:
                left += 1
            elif cur_sum > target:
                right -= 1
            else:
                return [left+1, right+1]
        return [-1, -1]
```

## Find Smallest Letter Greater Than Target

```py
'''
Find Smallest Letter Greater Than Target
Easy

Given a characters array letters that is sorted in non-decreasing order and a character target, return the smallest character in the array that is larger than target.

Note that the letters wrap around.

    For example, if target == 'z' and letters == ['a', 'b'], the answer is 'a'.

 

Example 1:

Input: letters = ["c","f","j"], target = "a"
Output: "c"

Example 2:

Input: letters = ["c","f","j"], target = "c"
Output: "f"

Example 3:

Input: letters = ["c","f","j"], target = "d"
Output: "f"

Example 4:

Input: letters = ["c","f","j"], target = "g"
Output: "j"

Example 5:

Input: letters = ["c","f","j"], target = "j"
Output: "c"
'''


class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        word_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 
                       'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 
                       'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
        
        target_num = word_to_num[target]
                
        for char in letters:
            if word_to_num[char] > target_num:
                return char
        return letters[0]
            

```

## Valid Parentheses

```py
'''
Valid Parentheses
Easy

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.

'''

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        chars = [char for char in s]
        
        for c in chars:
            if c == '(' or c == '[' or c == '{':
                stack.append(c)
            elif stack and c == ')' and stack[-1:][0] == '(':
                stack.pop()
            elif stack and c == ']' and stack[-1:][0] == '[':
                stack.pop()
            elif stack and c == '}' and stack[-1:][0] == '{':
                stack.pop()
            else:
                return False
        return True if len(stack) == 0 else False

```

## Happy Number

```py
'''
Happy Number
Easy

Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

    Starting with any positive integer, replace the number by the sum of the squares of its digits.
    Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
    Those numbers for which this process ends in 1 are happy.

Return true if n is a happy number, and false if not.

 

Example 1:

Input: n = 19
Output: true
Explanation:
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

Example 2:

Input: n = 2
Output: false
'''

class Solution:
    def isHappy(self, n: int) -> bool:
        while n != 1:
            digits = [int(char) ** 2 for char in str(n)]
            n = sum(digits)
            if n == 4:
                return False
        return True

```

## Largest Number At Least Twice of Others

```py
'''
Largest Number At Least Twice of Others
Easy

You are given an integer array nums where the largest integer is unique.

Determine whether the largest element in the array is at least twice as much as every other number in the array. If it is, return the index of the largest element, or return -1 otherwise.

 

Example 1:

Input: nums = [3,6,1,0]
Output: 1
Explanation: 6 is the largest integer.
For every other number in the array x, 6 is at least twice as big as x.
The index of value 6 is 1, so we return 1.

Example 2:

Input: nums = [1,2,3,4]
Output: -1
Explanation: 4 is less than twice the value of 3, so we return -1.

Example 3:

Input: nums = [1]
Output: 0
Explanation: 1 is trivially at least twice the value as any other number because there are no other numbers.
'''


class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        max_num = max(nums)
        max_index = nums.index(max_num)
        nums.remove(max_num)
        if len(nums) == 0:
            return 0
        second_max = max(nums)
        
        if max_num >= (2 * second_max):
            return max_index
        
        return -1

```

## Range Sum of BST

```py
'''
Range Sum of BST
Easy

Given the root node of a binary search tree and two integers low and high, return the sum of values of all nodes with a value in the inclusive range [low, high].

 

Example 1:

Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
Output: 32
Explanation: Nodes 7, 10, and 15 are in the range [7, 15]. 7 + 10 + 15 = 32.

Example 2:

Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
Output: 23
Explanation: Nodes 6, 7, and 10 are in the range [6, 10]. 6 + 7 + 10 = 23.
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        res = [0]
        
        self.inorder(root, low, high, res)
        
        return res[0]
    
    def inorder(self, root, low, high, res):
        if root == None:
            return
            
        # ignore all left elements
        if root.val < low:
            self.inorder(root.right, low, high, res)
        
        # ignore all right elements
        elif root.val > high:
            self.inorder(root.left, low, high, res)
            
        else:
            res[0] += root.val
            self.inorder(root.left, low, high, res)
            self.inorder(root.right, low, high, res)

```

## Majority Element

```py
'''
Majority Element
Easy

Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.
'''

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        memo = {}
        for a in nums:
            if a in memo:
                memo[a] += 1
            else:
                memo[a] = 1
        max_val = 0
        max_ele = nums[0]
        for key, value in memo.items():
            if value > max_val:
                max_ele = key
                max_val = value
        return max_ele
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

## Range Sum Query-Immutable

```py
'''
Range Sum Query - Immutable
Easy

Given an integer array nums, handle multiple queries of the following type:

    Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.

Implement the NumArray class:

    NumArray(int[] nums) Initializes the object with the integer array nums.
    int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).

'''

class NumArray:
    nums = []
    def __init__(self, nums: List[int]):
        self.nums = nums.copy()

    def sumRange(self, left: int, right: int) -> int:
        return sum(self.nums[left:right+1])


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)

```

## Increasing Order Search Tree

```py
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

## Sqrt x

```py
'''
Sqrt(x)
Easy

Given a non-negative integer x, compute and return the square root of x.

Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

Note: You are not allowed to use any built-in exponent function or operator, such as pow(x, 0.5) or x ** 0.5.
'''

class Solution:
    def mySqrt(self, x: int) -> int:
        i = 1
        while i*i <= x:
            if i*i == x:
                return i
            i += 1
        return i - 1
```

## Detect Capital

```py
'''
Detect Capital
Easy

We define the usage of capitals in a word to be right when one of the following cases holds:

    All letters in this word are capitals, like "USA".
    All letters in this word are not capitals, like "leetcode".
    Only the first letter in this word is capital, like "Google".

Given a string word, return true if the usage of capitals in it is right.
'''

class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if word.isupper() or word.islower():
            return True
        
        word = word[1:]
        
        if (word.islower()):
            return True
        
        return False
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

## Find All Duplicates in an Array

```py
'''
Find All Duplicates in an Array
Medium

Given an integer array nums of length n where all the integers of nums are in the range [1, n] and each integer appears once or twice, return an array of all the integers that appears twice.

You must write an algorithm that runs in O(n) time and uses only constant extra space.
'''

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        dic = {}
        res = []
        for i in nums:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
        for i in nums:
            if i in dic and dic[i] > 1:
                res.append(i)
                dic.pop(i)
        return res
```

## Find First and Last Position of Element in Sorted Array

```py
'''
Find First and Last Position of Element in Sorted Array
Medium

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.
'''

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        res = [-1, -1]
        res[0] = self.starting(nums, target)
        res[1] = self.ending(nums, target)
        return res
    
    def starting(self, nums, target):
        index = -1
        start = 0
        end = len(nums) - 1
        
        while start <= end:
            mid = int(start + (end - start) / 2)
            if nums[mid] >= target:
                end = mid - 1
            else:
                start = mid + 1
                
            if nums[mid] == target:
                index = mid
                
        return index
    
    def ending(self, nums, target):
        index = -1
        start = 0
        end = len(nums) - 1
        
        while start <= end:
            mid = int(start + (end - start) / 2)
            if nums[mid] <= target:
                start = mid + 1
            else:
                end = mid - 1
                
            if nums[mid] == target:
                index = mid
                
        return index
```

## Find the Highest Altitude

```py
'''
Find the Highest Altitude
Easy

There is a biker going on a road trip. The road trip consists of n + 1 points at different altitudes. The biker starts his trip on point 0 with altitude equal 0.

You are given an integer array gain of length n where gain[i] is the net gain in altitude between points i​​​​​​ and i + 1 for all (0 <= i < n). Return the highest altitude of a point.
'''

class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        net_alt = [0, gain[0]]
        
        for i in range(1, len(gain)):
            net_alt.append(sum(gain[:i+1]))
            
        return max(net_alt)
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

## Squares of a Sorted Array

```py
'''
Squares of a Sorted Array
Easy

Given an integer array nums sorted in non-decreasing order, 
return an array of the squares of each number sorted in non-decreasing order.
'''

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        negs = []
        pos = []
        for i in nums:
            if i < 0:
                negs.insert(0, i ** 2)
            else:
                pos.append(i ** 2)
        res = []
        l1 = 0
        l2 = 0
        while l1 < len(negs) and l2 < len(pos):
            if negs[l1] < pos[l2]:
                res.append(negs[l1])
                l1 += 1
            else:
                res.append(pos[l2])
                l2 += 1
        while l1 < len(negs):
            res.append(negs[l1])
            l1 += 1
        while l2 < len(pos):
            res.append(pos[l2])
            l2 += 1
        return res
```

## Add Strings

```py
'''
Add Strings
Easy

Given two non-negative integers, num1 and num2 represented as string, return the sum of num1 and num2 as a string.

You must solve the problem without using any built-in library for handling large integers (such as BigInteger). You must also not convert the inputs to integers directly.
'''

class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        return str(int(num1) + int(num2))
```

## Fibonacci Number

```py
'''
Fibonacci Number
Easy

The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,

F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.

Given n, calculate F(n).
'''

class Solution:
    def fib(self, n: int, memo={}) -> int:
        if n == 0:
            return 0
        if n <= 2:
            return 1
        if n in memo:
            return memo[n]
        else:
            memo[n] = self.fib(n-1, memo) + self.fib(n-2, memo)
            return memo[n]
```

## First Missing Positive

```py
'''
First Missing Positive
Hard

Given an unsorted integer array nums, find the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses constant extra space.
'''

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        min_pos = 1
        memo = {}
        for i, a in enumerate(nums):
            memo[a] = True
            if a > 0 and a == min_pos:
                min_pos += 1
        while min_pos in memo:
            min_pos += 1
        return min_pos
```

## Remove Duplicates from Sorted Array II

```py
'''
Remove Duplicates from Sorted Array II
Medium

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}

If all assertions pass, then your solution will be accepted.

 

Example 1:

Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).

Example 2:

Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
Explanation: Your function should return k = 7, with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
'''

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)
        right = len(nums) - 1
        
        left = 2
        
        while left <= right:
            if nums[left-2] == nums[left-1] and nums[left-1] == nums[left]:
                temp = nums[left]
                nums.remove(nums[left])
                nums.append(temp)
                right -= 1
            else:
                left += 1
                
        return left

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

## Merge Intervals

```py
'''
Merge Intervals
Medium

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that 
cover all the intervals in the input.
'''

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = []
        
        while self.has_interval(intervals):
            i = 0
            res = []
            while i < len(intervals):
                try:
                    if intervals[i][1] >= intervals[i+1][0]:
                        res.append([min(intervals[i][0], intervals[i+1][0]), max(intervals[i][1], intervals[i+1][1])])
                        i += 2
                    else:
                        res.append(intervals[i])
                        i += 1
                except IndexError:
                    res.append(intervals[i])
                    i += 1
            intervals = copy.copy(res)
        return res if len(res) > 0 else intervals
    
    def has_interval(self, intervals):
        for i in range(len(intervals)-1):
            if intervals[i][1] >= intervals[i+1][0]:
                return True
        return False

```

## Linked List Cycle

```py
'''
Linked List Cycle
Easy

Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if head == None:
            return False
        fast = head.next
        slow = head
        while fast != None and fast.next != None:
            if fast.next == slow:
                return True
            fast = fast.next.next
            slow = slow.next
        return False
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

## Power of Four

```py
'''
Power of Four
Easy

Given an integer n, return true if it is a power of four. Otherwise, return false.

An integer n is a power of four, if there exists an integer x such that n == 4x.
'''

class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1:
            return True
        if n < 4:
            return False
        while n > 1:
            if n % 4 != 0:
                return False
            n = int(n / 4)
        return True
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

## Single Number

```py
'''
Single Number
Easy

Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.
'''

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        memo  ={}
        for i in nums:
            if i in memo:
                memo[i] += 1
            else:
                memo[i] = 1
        for i in nums:
            if i in memo and memo[i] == 1:
                return i
        return -1
```

## Set Matrix Zeroes

```py
'''
Set Matrix Zeroes
Medium

Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's, and return the matrix.

You must do it in place.

 

Example 1:

Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:

Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
'''

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        res = []
        for i in range(len(matrix)):
            res.append(matrix[i].copy())
        
        for i in range(len(res)):
            for j in range(len(res[i])):
                if res[i][j] == 0:
                    for k in range(len(res)):
                        matrix[k][j] = 0
                    for k in range(len(res[i])):
                        matrix[i][k] = 0

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

## Sort Array By Parity

```py
'''
Sort Array By Parity
Easy

Given an array nums of non-negative integers, return an array consisting of all the even elements of nums, followed by all the odd elements of nums.

You may return any answer array that satisfies this condition.
'''

class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        res = []
        
        for i in A:
            if i % 2 == 0:
                res.append(i)
                
        for i in A:
            if i % 2 == 1:
                res.append(i)
                
        return res
```

## Peak Index in a Mountain Array

```py
'''
Peak Index in a Mountain Array
Easy

Let's call an array arr a mountain if the following properties hold:

    arr.length >= 3
    There exists some i with 0 < i < arr.length - 1 such that:
        arr[0] < arr[1] < ... arr[i-1] < arr[i]
        arr[i] > arr[i+1] > ... > arr[arr.length - 1]

Given an integer array arr that is guaranteed to be a mountain, return any i such that arr[0] < arr[1] < ... arr[i - 1] < arr[i] > arr[i + 1] > ... > arr[arr.length - 1].
'''

class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        return arr.index(max(arr))
```

## Remove Duplicates from Sorted Array

```py
'''
Remove Duplicates from Sorted Array
Easy

Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
'''

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        res = 0
        
        for i in nums:
            while nums.count(i) > 1:
                nums.remove(i)
        return len(nums)
```

## Minimum Index Sum of Two Lists

```py
'''
Minimum Index Sum of Two Lists
Easy

Suppose Andy and Doris want to choose a restaurant for dinner, and they both have a list of favorite restaurants represented by strings.

You need to help them find out their common interest with the least list index sum. If there is a choice tie between answers, output all of them with no order requirement. You could assume there always exists an answer.

 

Example 1:

Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["Piatti","The Grill at Torrey Pines","Hungry Hunter Steakhouse","Shogun"]
Output: ["Shogun"]
Explanation: The only restaurant they both like is "Shogun".

Example 2:

Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["KFC","Shogun","Burger King"]
Output: ["Shogun"]
Explanation: The restaurant they both like and have the least index sum is "Shogun" with index sum 1 (0+1).

Example 3:

Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["KFC","Burger King","Tapioca Express","Shogun"]
Output: ["KFC","Burger King","Tapioca Express","Shogun"]

Example 4:

Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["KNN","KFC","Burger King","Tapioca Express","Shogun"]
Output: ["KFC","Burger King","Tapioca Express","Shogun"]

Example 5:

Input: list1 = ["KFC"], list2 = ["KFC"]
Output: ["KFC"]

'''


class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        dic = {}
        
        for i, word1 in enumerate(list1):
            for j, word2 in enumerate(list2):
                if word1 == word2:
                    dic[word1] = i + j
        res = []
        min_val = len(list1) + len(list2)
        for (key, value) in dic.items():
            if value < min_val:
                min_val = value
                res = [key]
            elif min_val == value:
                res.append(key)
        return res

```

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

## Determine if String Halves Are Alike

```py
'''
Determine if String Halves Are Alike
Easy

You are given a string s of even length. Split this string into two halves of equal lengths, and let a be the first half and b be the second half.

Two strings are alike if they have the same number of vowels ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'). Notice that s contains uppercase and lowercase letters.

Return true if a and b are alike. Otherwise, return false.
'''

class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        s1 = s[:int(len(s) / 2)]
        s2 = s[int(len(s) / 2):]
        
        c1 = s1.count("a")
        c1 += s1.count("e")
        c1 += s1.count("i")
        c1 += s1.count("o")
        c1 += s1.count("u")
        
        c1 += s1.count("A")
        c1 += s1.count("E")
        c1 += s1.count("I")
        c1 += s1.count("O")
        c1 += s1.count("U")
        
        c2 = s2.count("a")
        c2 += s2.count("e")
        c2 += s2.count("i")
        c2 += s2.count("o")
        c2 += s2.count("u")
                       
        c2 +=s2.count("A")
        c2 += s2.count("E")
        c2 += s2.count("I")
        c2 += s2.count("O")
        c2 += s2.count("U")
        
        return (True if c1 == c2 else False)
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

## Power of Three

```py
'''
Power of Three
Easy

Given an integer n, return true if it is a power of three. Otherwise, return false.

An integer n is a power of three, if there exists an integer x such that n == 3x.
'''

class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1:
            return True
        if n < 3:
            return False
        while n > 1:
            if n % 3 != 0:
                return False
            n = int(n / 3)
        return True
```

## Minimum Depth of Binary Tree

```py
'''
Minimum Depth of Binary Tree
Easy

Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0
        if root.left == None:
            return 1 + self.minDepth(root.right)
        if root.right == None:
            return 1 + self.minDepth(root.left)
        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
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

## Number of Islands

```py
'''
Number of Islands
Medium

Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
'''

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if grid == None or len(grid) == 0:
            return 0
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '1':
                    count += self.dfs(grid, i, j)
        return count
    
    def dfs(self, grid, i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[i]) or grid[i][j] == '0':
            return 0
        grid[i][j] = '0'
        self.dfs(grid, i+1, j)
        self.dfs(grid, i-1, j)
        self.dfs(grid, i, j+1)
        self.dfs(grid, i, j-1)
        return 1
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

## Contains Duplicate II

```py
'''
Contains Duplicate II
Easy

Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.
'''

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dups = {}
        for i in range(len(nums)):
            if nums[i] in dups:
                for j in range(len(nums)):
                    if i != j and nums[i] == nums[j] and abs(i-j) <= k:
                        return True
            else:
                dups[nums[i]] = i
                    
        return False
```

## Remove Nth Node From End of List

```py
'''
Remove Nth Node From End of List
Medium

Given the head of a linked list, remove the nth node from the end of the list and return its head.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        temp = ListNode()
        temp.next = head
        
        slow = temp
        fast = temp
        
        for _ in range(n+1):
            fast = fast.next
        
        while fast != None:
            fast = fast.next
            slow = slow.next
        
        slow.next = slow.next.next
        return temp.next

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

## Remove Linked List Elements

```py
'''
Remove Linked List Elements
Easy

Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        cur = head
        
        while cur != None and cur.next != None:
            while cur.next != None and cur.next.val == val:
                cur.next = cur.next.next
            cur = cur.next
            
        if head != None and head.val == val:
            head = head.next
            return head
        return head
```

## N-th Tribonacci Number

```py
'''
N-th Tribonacci Number
Easy

The Tribonacci sequence Tn is defined as follows: 

T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

Given n, return the value of Tn.
'''

class Solution:
    def tribonacci(self, n: int, memo={}) -> int:
        if n == 0:
            return 0
        if n <= 2:
            return 1
        if n in memo:
            return memo[n]
        else:
            memo[n] = self.tribonacci(n-1, memo) + self.tribonacci(n-2, memo) + self.tribonacci(n-3, memo)
            return memo[n]
```

## Rotate List

```py
'''
Rotate List
Medium

Given the head of a linked list, rotate the list to the right by k places.

 

Example 1:

Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Example 2:

Input: head = [0,1,2], k = 4
Output: [2,0,1]
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head == None or head.next == None:
            return head
        size = 0
        cur = head
        
        # to find the size of the list to reduce unnecessary roatations
        while cur != None:
            size += 1
            cur = cur.next
        
        # running the loop for only required size which is (k mod size)
        for _ in range(k % size):
            fast = head
            prev = fast
            while fast.next != None:
                prev = fast
                fast = fast.next
            prev.next = None
            fast.next = head
            head = fast
        return head

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

## String Matching in an Array

```js
/*
String Matching in an Array
Easy

Given an array of string words. Return all strings in words which is substring of another word in any order. 

String words[i] is substring of words[j], if can be obtained removing some characters to left and/or right side of words[j].
*/

/**
 * @param {string[]} words
 * @return {string[]}
 */
var stringMatching = function (words) {
    res = [];
    for (word1 of words) {
        const re = new RegExp(word1, "gim");
        for (word2 of words) {
            if (word1 != word2 && re.test(word2) && !res.includes(word1)) {
                res.push(word1);
            }
        }
    }
    return res;
};

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

## Subarray Product Less Than K

```py
'''
Subarray Product Less Than K
Medium

Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.
'''

class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0
        left = 0
        right = 0
        prod = 1
        count = 0
        while right < len(nums):
            prod = prod * nums[right]
            while prod >= k:
                prod /= nums[left]
                left += 1
            count += right - left + 1
            right += 1
        return count
```

## Implement strStr

```py
'''
Implement strStr()
Easy

Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Clarification:

What should we return when needle is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr() and Java's indexOf().
'''

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == '':
            return 0
        i, j = 0, len(needle) - 1
        while j < len(haystack):
            if haystack[i:j+1] != needle:
                i += 1
                j += 1
            else:
                return i
        return -1

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

## Check if the Sentence Is Pangram

```py
'''
Check if the Sentence Is Pangram
Easy

A pangram is a sentence where every letter of the English alphabet appears at least once.

Given a string sentence containing only lowercase English letters, return true if sentence is a pangram, or false otherwise.
'''

class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        if len(sentence) < 26:
            return False
        alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        for char in alphabets:
            if sentence.count(char) < 1:
                return False
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

## Unique Paths

```py
'''
Unique Paths
Medium

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
'''

class Solution:
    def uniquePaths(self, m: int, n: int, memo={}) -> int:
        key = f'{m},{n}'
        if key in memo:
            return memo[key]
        if m == 1 and n == 1:
            return 1
        if m == 0 or n == 0:
            return 0
        memo[key] = self.uniquePaths(m-1, n, memo) + self.uniquePaths(m, n-1, memo)
        return memo[key]
```

## Find Peak Element

```py
'''
Find Peak Element
Medium

A peak element is an element that is strictly greater than its neighbors.

Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞.

You must write an algorithm that runs in O(log n) time.
'''

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        # if len(nums) < 2:
        #     return 0
        # if len(nums) == 2:
        #     return nums.index(max(nums))
        # left, right = 0, len(nums)-1
        # while left <= right:
        #     mid = int(left + (right-left) / 2)
        #     if nums[mid-1] and nums[mid+1] and nums[mid] > nums[mid-1] and nums[mid] > nums[mid+1]:
        #         return mid
        #     elif nums[mid] < nums[mid-1]:
        #         right = mid -1
        #     elif nums[mid] < nums[mid+1]:
        #         left = mid + 1
        # return -1
        return nums.index(max(nums))
```

## Binary Tree Inorder Traversal

```py
'''
Binary Tree Inorder Traversal
Easy

Given the root of a binary tree, return the inorder traversal of its nodes' values.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        
        self.inorder(root, res)
        
        return res
    
    def inorder(self, root, res):
        if root == None:
            return
        
        self.inorder(root.left, res)
        res.append(root.val)
        self.inorder(root.right, res)
```

## Rotting Oranges

```py
'''
Rotting Oranges
Medium

You are given an m x n grid where each cell can have one of three values:

    0 representing an empty cell,
    1 representing a fresh orange, or
    2 representing a rotten orange.

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.

 

Example 1:

Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4

Example 2:

Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.

Example 3:

Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
'''

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        temp = []
        minutes = 0
        for i in range(len(grid)):
            temp.append(grid[i].copy())
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 2:
                    if i > 0 and temp[i-1][j] == 1:
                        temp[i-1][j] = 2
                    if i < len(grid) - 1 and temp[i+1][j] == 1:
                        temp[i+1][j] = 2
                    
                    if j > 0 and temp[i][j-1] == 1:
                        temp[i][j-1] = 2
                    if j < len(grid[i]) - 1 and temp[i][j+1] == 1:
                        temp[i][j+1] = 2
        minutes += 1
        
        while temp != grid:
            grid = []
            for i in range(len(temp)):
                grid.append(temp[i].copy())
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == 2:
                        if i > 0 and temp[i-1][j] == 1:
                            temp[i-1][j] = 2
                        if i < len(grid) - 1 and temp[i+1][j] == 1:
                            temp[i+1][j] = 2

                        if j > 0 and temp[i][j-1] == 1:
                            temp[i][j-1] = 2
                        if j < len(grid[i]) - 1 and temp[i][j+1] == 1:
                            temp[i][j+1] = 2
            minutes += 1
            
        print(temp)
            
        for i in range(len(temp)):
            if temp[i].count(1) > 0:
                return -1
            
        return minutes - 1

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

## Reverse Linked List

```py
'''
Reverse Linked List
Easy

Given the head of a singly linked list, reverse the list, and return the reversed list.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        
        while (head != None):
            lnext = head.next
            head.next = prev
            prev = head
            head = lnext
            
        return prev
```

## Sort Colors

```py
'''
Sort Colors
Medium

Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.
'''

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        self.quickSort(nums, 0, len(nums)-1)
        
    def quickSort(self, nums, low, high):
        if len(nums) == 1:
            return nums
        if low < high:
            pi = self.partition(nums, low, high)

            self.quickSort(nums, low, pi-1)
            self.quickSort(nums, pi+1, high)
            
    def partition(self, nums, low, high):
        i = (low-1)
        pivot = nums[high]

        for j in range(low, high):
            if nums[j] <= pivot:
                i = i+1
                nums[i], nums[j] = nums[j], nums[i]

        nums[i+1], nums[high] = nums[high], nums[i+1]
        return (i+1)
```

## Shuffle the Array

```js
/*
Shuffle the Array
Easy

Given the array nums consisting of 2n elements in the form [x1,x2,...,xn,y1,y2,...,yn].

Return the array in the form [x1,y1,x2,y2,...,xn,yn].
*/

/**
 * @param {number[]} nums
 * @param {number} n
 * @return {number[]}
 */
var shuffle = function (nums, n) {
    let res = [];
    for (let i = 0; i < n; i++) {
        res.push(...[nums[i], nums[i + n]]);
    }
    return res;
};

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

## Maximum Average Subarray I

```py
'''
Maximum Average Subarray I
Easy

You are given an integer array nums consisting of n elements, and an integer k.

Find a contiguous subarray whose length is equal to k that has the maximum average value and return this value. Any answer with a calculation error less than 10-5 will be accepted.

 

Example 1:

Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75

Example 2:

Input: nums = [5], k = 1
Output: 5.00000
'''


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        if len(nums) == 1:
            return float(nums[0])
        left = 0
        right = k - 1
        cur_sum = sum(nums[left:right])
        max_avg = (cur_sum + nums[right]) / k
        
        while right < len(nums):
            cur_sum += nums[right]
            max_avg = max(max_avg, cur_sum / k)
            
            cur_sum -= nums[left]
            
            left += 1
            right += 1
            
        return max_avg

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

## Rotate Image

```py
'''
Rotate Image
Medium

You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:

Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]

Example 3:

Input: matrix = [[1]]
Output: [[1]]

Example 4:

Input: matrix = [[1,2],[3,4]]
Output: [[3,1],[4,2]]

'''

# Intuition is to take transpose and then reverse each row


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        res = []
        for i in range(len(matrix)):
            res.append(matrix[i].copy())
        
        for i in range(len(res)):
            for j in range(len(res[i])):
                matrix[i][j] = res[j][i]
            matrix[i].reverse()
            
            

```

## Median of Two Sorted Arrays

```js
/*
Median of Two Sorted Arrays
Hard

Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).
*/

/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number}
 */
var findMedianSortedArrays = function (nums1, nums2) {
    let newArr = [...nums1, ...nums2];
    newArr.sort((a, b) => {
        return a - b;
    });

    //     if (nums1.length == 0 && nums2.length != 0) {
    //         newArr = nums2;
    //     }

    //     if (nums2.length == 0 && nums1.length != 0) {
    //         newArr = nums1;
    //     }

    if (newArr.length % 2 != 0) {
        return newArr[Math.floor(newArr.length / 2)];
    } else {
        return (newArr[newArr.length / 2] + newArr[newArr.length / 2 - 1]) / 2;
    }
};

```

## Maximum Subarray

```py
'''
Maximum Subarray
Easy

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
'''

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        cur_sum = max_sum
        for i in range(1, len(nums)):
            cur_sum = max(nums[i] + cur_sum, nums[i])
            max_sum = max(cur_sum, max_sum)
        return max_sum
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

## Two Sum

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

## Validate Binary Search Tree

```py
'''
Validate Binary Search Tree
Medium

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

    The left subtree of a node contains only nodes with keys less than the node's key.
    The right subtree of a node contains only nodes with keys greater than the node's key.
    Both the left and right subtrees must also be binary search trees.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.isValid(root, None, None)
    
    def isValid(self, root, max_val, min_val):
        if root == None:
            return True
        elif max_val != None and root.val >= max_val or min_val != None and root.val <= min_val:
            return False
        else:
            return self.isValid(root.left, root.val, min_val) and self.isValid(root.right, max_val, root.val) 

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

## longest substring without repeating chars

```py
"""
Longest Substring Without Repeating Characters
Medium

Given a string s, find the length of the longest substring without repeating characters.
"""

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left = 0
        right = 0
        
        max_len = 0
        memo = {}
        
        while right < len(s):
            if s[right] in memo:
                memo.pop(s[left])
                left += 1
                continue
            
            memo[s[right]] = True
            max_len = max(max_len, len(s[left:right+1]))
            right += 1
            
            
        return max_len;
```

## Length of Last Word

```py
'''
Length of Last Word
Easy

Given a string s consists of some words separated by spaces, return the length of the last word in the string. If the last word does not exist, return 0.

A word is a maximal substring consisting of non-space characters only.
'''

class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        words = s.split(" ")
        words = [word for word in words if word != ""]
        try:
            last_word = words[-1]
            return len(last_word)
        except IndexError:
            return 0

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

## Best Time to Buy and Sell Stock

```py
'''
Best Time to Buy and Sell Stock
Easy

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
'''

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        min_price = prices[0]
        for price in prices:
            min_price = min(price, min_price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

```

## Sum of All Odd Length Subarrays

```py
'''
 Sum of All Odd Length Subarrays
Easy

Given an array of positive integers arr, calculate the sum of all possible odd-length subarrays.

A subarray is a contiguous subsequence of the array.

Return the sum of all odd-length subarrays of arr.

 

Example 1:

Input: arr = [1,4,2,5,3]
Output: 58
Explanation: The odd-length subarrays of arr and their sums are:
[1] = 1
[4] = 4
[2] = 2
[5] = 5
[3] = 3
[1,4,2] = 7
[4,2,5] = 11
[2,5,3] = 10
[1,4,2,5,3] = 15
If we add all these together we get 1 + 4 + 2 + 5 + 3 + 7 + 11 + 10 + 15 = 58

Example 2:

Input: arr = [1,2]
Output: 3
Explanation: There are only 2 subarrays of odd length, [1] and [2]. Their sum is 3.

Example 3:

Input: arr = [10,11,12]
Output: 66
'''


class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        res = 0
        for i in range(0, len(arr)):
            for j in range(i, len(arr)):
                if len(arr[i:j+1]) % 2 == 1:
                    res += sum(arr[i:j+1])
                    
        return res

```

## Remove Element

```py
'''
Remove Element
Easy

Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
'''

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        while nums.count(val) > 0:
            nums.remove(val)
        return len(nums)
```

## Path Sum

```py
'''
Path Sum
Easy

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if root == None:
            return False
        if root.val == targetSum and root.left == None and root.right == None:
            return True
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)
```

## Add Binary

```py
'''
Add Binary
Easy

Given two binary strings a and b, return their sum as a binary string.
'''

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        res = int(a, 2) + int(b, 2)
        
        return bin(res)[2:]
```

## Lucky Numbers in a Matrix

```py
'''
Lucky Numbers in a Matrix
Easy

Given a m * n matrix of distinct numbers, return all lucky numbers in the matrix in any order.

A lucky number is an element of the matrix such that it is the minimum element in its row and maximum in its column.
'''

class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        res = []
        
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if self.isLucky(matrix[i][j], matrix[i], self.getCols(j, matrix), matrix):
                    res.append(matrix[i][j])
                    
        return res
                    
    def getCols(self, column, mat):
        col = []
        
        for i in range(len(mat)):
            col.append(mat[i][column])
            
        return col
                
    
    def isLucky(self, num, row, col, mat):
        if num == min(row) and num == max(col):
            return True
        else:
            return False
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

## Defanging an IP Address

```py
'''
Defanging an IP Address
Easy

Given a valid (IPv4) IP address, return a defanged version of that IP address.

A defanged IP address replaces every period "." with "[.]".
'''

class Solution:
    def defangIPaddr(self, address: str) -> str:
        return address.replace(".", "[.]")
```

## Find All Numbers Disappeared in an Array

```py
'''
Find All Numbers Disappeared in an Array
Easy

Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers in the range [1, n] that do not appear in nums.
'''

class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        res = []
        memo = {}
        for i in nums:
            memo[i] = True
        for i in range(1, len(nums)+1):
            if i not in memo:
                res.append(i)
        return res
```

## Same Tree

```py
'''
Same Tree
Easy

Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p == None and q == None:
            return True
        if (p == None and q != None) or (p != None and q == None):
            return False
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False
```

## Search in Rotated Sorted Array

```py
'''
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:

Input: nums = [1], target = 0
Output: -1
'''

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
                
        start = left
        left, right = 0, len(nums) - 1
        
        if target >= nums[start] and target <= nums[right]:
            left = start
        else:
            right = start
            
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
                
        return -1

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

## Word Search

```py
'''
Word Search
Medium

Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Example 2:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Example 3:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
'''

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == word[0] and self.dfs(board, i, j, 0, word):
                    return True
        
        return False
    
    def dfs(self, board, i, j, count, word):
        if count == len(word):
            return True
        
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[i]) or board[i][j] != word[count]:
            return False
        
        temp = board[i][j]
        board[i][j] = " "
        
        found = self.dfs(board, i+1, j, count+1, word) or self.dfs(board, i-1, j, count+1, word) or self.dfs(board, i, j+1, count+1, word) or self.dfs(board, i, j-1, count+1, word)
        
        board[i][j] = temp
        return found
        

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

## Remove All Occurrences of a Substring

```py
'''
Remove All Occurrences of a Substring
Medium

Given two strings s and part, perform the following operation on s until all occurrences of the substring part are removed:

    Find the leftmost occurrence of the substring part and remove it from s.

Return s after removing all occurrences of part.

A substring is a contiguous sequence of characters in a string.

 

Example 1:

Input: s = "daabcbaabcbc", part = "abc"
Output: "dab"
Explanation: The following operations are done:
- s = "daabcbaabcbc", remove "abc" starting at index 2, so s = "dabaabcbc".
- s = "dabaabcbc", remove "abc" starting at index 4, so s = "dababc".
- s = "dababc", remove "abc" starting at index 3, so s = "dab".
Now s has no occurrences of "abc".

Example 2:

Input: s = "axxxxyyyyb", part = "xy"
Output: "ab"
Explanation: The following operations are done:
- s = "axxxxyyyyb", remove "xy" starting at index 4 so s = "axxxyyyb".
- s = "axxxyyyb", remove "xy" starting at index 3 so s = "axxyyb".
- s = "axxyyb", remove "xy" starting at index 2 so s = "axyb".
- s = "axyb", remove "xy" starting at index 1 so s = "ab".
Now s has no occurrences of "xy".
'''


class Solution:
    def removeOccurrences(self, s: str, part: str) -> str:
        while part in s:
            s = s.replace(part, '', 1)
        return s

```

## Sort an Array

```py
'''
Sort an Array
Medium

Given an array of integers nums, sort the array in ascending order.
'''

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.quicksort(nums, 0, len(nums) - 1);
        return nums
        
    def quicksort(self, a, first, last):
        i = 0
        j = 0
        pivot = 0

        if first < last:
            pivot = first
            i = first
            j = last

            while (i < j):
                while a[i] <= a[pivot] and i < last:
                    i += 1
                while a[j] > a[pivot]:
                    j -= 1

                if i < j:
                    temp = a[i]
                    a[i] = a[j]
                    a[j] = temp

            temp = a[pivot]
            a[pivot] = a[j]
            a[j] = temp

            self.quicksort(a, first, j - 1);
            self.quicksort(a, j + 1, last);
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

## Binary Tree Paths

```py
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

```

## Reverse String

```py
'''
Reverse String
Easy

Write a function that reverses a string. The input string is given as an array of characters s.
'''

class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        s.reverse()
```

## Keys and Rooms

```py
'''
Keys and Rooms
Medium

There are N rooms and you start in room 0.  Each room has a distinct number in 0, 1, 2, ..., N-1, and each room may have some keys to access the next room. 

Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., N-1] where N = rooms.length.  A key rooms[i][j] = v opens the room with number v.

Initially, all the rooms start locked (except for room 0). 

You can walk back and forth between rooms freely.

Return true if and only if you can enter every room.
'''

class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        seen = [False for r in rooms]
        seen[0] = True
        
        keys = []
        keys.append(0)
        
        while len(keys) != 0:
            cur = keys.pop()
            for new_key in rooms[cur]:
                if not seen[new_key]:
                    seen[new_key] = True
                    keys.append(new_key)
                    
        if False in seen:
            return False
        return True
```

## Middle of the Linked List

```py
'''
Middle of the Linked List
Easy

Given a non-empty, singly linked list with head node head, return a middle node of linked list.

If there are two middle nodes, return the second middle node.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            
        return slow
```

## Shuffle an Array

```py
'''
Shuffle an Array
Medium

Given an integer array nums, design an algorithm to randomly shuffle the array. All permutations of the array should be equally likely as a result of the shuffling.

Implement the Solution class:

- Solution(int[] nums) Initializes the object with the integer array nums.
- int[] reset() Resets the array to its original configuration and returns it.
- int[] shuffle() Returns a random shuffling of the array.

'''

import random
class Solution:
    arr = []

    def __init__(self, nums: List[int]):
        self.arr = nums.copy()

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        return self.arr
        

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        temp = self.arr.copy()
        random.shuffle(temp)
        return temp
        
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()
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

## Search a 2D Matrix

```py
'''
Search a 2D Matrix
Medium

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the previous row.

'''

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        last_ind = len(matrix[0]) - 1
        row = -1
        for i in range(len(matrix)):
            if target <= matrix[i][last_ind]:
                row = i
                break
        if row == -1:
            return False
        for ele in matrix[row]:
            if ele == target:
                return True
        return False
```

## palindrome number

```py
'''
Palindrome Number
Easy

Given an integer x, return true if x is palindrome integer.

An integer is a palindrome when it reads the same backward as forward. For example, 121 is palindrome while 123 is not.
'''

class Solution:
    def isPalindrome(self, x: int) -> bool:
        num = str(x)        
        if num == num[::-1]:
            return True
        else:
            return False
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

## First Unique Character in a String

```py
'''
First Unique Character in a String
Easy

Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.
'''

class Solution:
    def firstUniqChar(self, s: str) -> int:
        words = {}
        
        for char in s:
            try:
                words[char] = words[char] + 1
            except Exception:
                words.update({char: 1})
                        
        for (key, value) in words.items():
            if value == 1:
                return s.index(key)
            
        return -1
```

## Second Largest Digit in a String

```py
'''
Second Largest Digit in a String
Easy

Given an alphanumeric string s, return the second largest numerical digit that appears in s, or -1 if it does not exist.

An alphanumeric string is a string consisting of lowercase English letters and digits.

 

Example 1:

Input: s = "dfa12321afd"
Output: 2
Explanation: The digits that appear in s are [1, 2, 3]. The second largest digit is 2.

Example 2:

Input: s = "abc1111"
Output: -1
Explanation: The digits that appear in s are [1]. There is no second largest digit. 
'''

class Solution:
    def secondHighest(self, s: str) -> int:
        max_num = 0
        
        for char in s:
            try:
                max_num = max(max_num, int(char))
            except ValueError:
                pass
            
        s = s.replace(str(max_num), "")
        max_num = -1
        
        for char in s:
            try:
                max_num = max(max_num, int(char))
            except ValueError:
                pass
            
        return max_num

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

## Binary Search

```py
'''
Binary Search
Easy

Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.
'''

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        return self.binary(nums, target, 0, len(nums) - 1)
    
    def binary(self, a, key, low, high):
        if low <= high:
            mid = int((low + high) / 2)
            
            if key == a[mid]:
                return mid
            
            elif key < a[mid]:
                return self.binary(a, key, low, mid - 1)
                
            else:
                return self.binary(a, key, mid + 1, high)
        else:
            return -1
```

## Delete Node in a Linked List

```py
'''
Delete Node in a Linked List
Easy

Write a function to delete a node in a singly-linked list. You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.

It is guaranteed that the node to be deleted is not a tail node in the list.
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
```

## Kth Smallest Element in a BST

```py
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
        
        

```

## Longest Common Prefix

```py
'''
Longest Common Prefix
Easy

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".
'''


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        min_len = len(strs[0])
        for word in strs:
            min_len = min(min_len, len(word))
        res = ""
        for ind in range(len(strs)):
            for i in range(min_len):
                if self.isEqual(strs, i):
                    res = strs[ind][:i+1]
        return res
    
    def isEqual(self, strs, index):
        for i in range(1, len(strs)):
            if strs[i-1][:index+1] != strs[i][:index+1]:
                return False
        return True
```

