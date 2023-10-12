
# leetcode programs:
## Page: 8
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

## Summary Ranges

```py
'''
Summary Ranges
Easy

You are given a sorted unique integer array nums.

Return the smallest sorted list of ranges that cover all the numbers in the array exactly. That is, each element of nums is covered by exactly one of the ranges, and there is no integer x such that x is in one of the ranges but not in nums.

Each range [a,b] in the list should be output as:

    "a->b" if a != b
    "a" if a == b

 

Example 1:

Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"

Example 2:

Input: nums = [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: The ranges are:
[0,0] --> "0"
[2,4] --> "2->4"
[6,6] --> "6"
[8,9] --> "8->9"

Example 3:

Input: nums = []
Output: []

Example 4:

Input: nums = [-1]
Output: ["-1"]

Example 5:

Input: nums = [0]
Output: ["0"]
'''


class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        res = []
        
        s = ""
        i = 0
        
        while i < len(nums):
            num1 = nums[i]
            
            while i < len(nums) - 1 and nums[i+1] == nums[i] + 1:
                i += 1
            if num1 == nums[i]:
                s = str(num1)
            else:
                s = f"{num1}->{nums[i]}"
            i += 1
            res.append(s)
            
        return res

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

# bottom up approach (DP)
class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        elif n < 2:
            return 1
            
        dp = [0] * (n + 1)
    
        dp[0] = 0
        dp[1] = 1
        dp[2] = 1

        if n < 3:
            return dp[n]

        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]
    
    
 

# recursive and memoized solution (DP)
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

## Simplified Fractions

```py
'''
Simplified Fractions
Medium

Given an integer n, return a list of all simplified fractions between 0 and 1 (exclusive) such that the denominator is less-than-or-equal-to n. The fractions can be in any order.

 

Example 1:

Input: n = 2
Output: ["1/2"]
Explanation: "1/2" is the only unique fraction with a denominator less-than-or-equal-to 2.

Example 2:

Input: n = 3
Output: ["1/2","1/3","2/3"]

Example 3:

Input: n = 4
Output: ["1/2","1/3","1/4","2/3","3/4"]
Explanation: "2/4" is not a simplified fraction because it can be simplified to "1/2".

Example 4:

Input: n = 1
Output: []
'''


class Solution:
    def simplifiedFractions(self, n: int) -> List[str]:
        res = []
        
        if n == 1:
            return res
        
        hs = set()
        
        for nume in range(1, n):
            for deno in range(2, n+1):
                t = nume / deno
                if deno > nume and t not in hs and t <= n:
                    res.append(f"{nume}/{deno}")
                    hs.add(t)
                    
        return res

```

## Self Dividing Numbers

```py
'''
Self Dividing Numbers
Easy

A self-dividing number is a number that is divisible by every digit it contains.

    For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.

A self-dividing number is not allowed to contain the digit zero.

Given two integers left and right, return a list of all the self-dividing numbers in the range [left, right].

 

Example 1:

Input: left = 1, right = 22
Output: [1,2,3,4,5,6,7,8,9,11,12,15,22]

Example 2:

Input: left = 47, right = 85
Output: [48,55,66,77]
'''


class Solution:
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        res = []
        
        for num in range(left, right+1):
            divs = [int(char) for char in str(num)]
            
            if 0 not in divs and self.divides(divs, num):
                res.append(num)
                
        return res
    
    def divides(self, divs, num):
        for n in divs:
            if num % n != 0:
                return False
            
        return True

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

## Next Greater Element I

```py
'''
Next Greater Element I
Easy

The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2. If there is no next greater element, then the answer for this query is -1.

Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

 

Example 1:

Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.

Example 2:

Input: nums1 = [2,4], nums2 = [1,2,3,4]
Output: [3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 2 is underlined in nums2 = [1,2,3,4]. The next greater element is 3.
- 4 is underlined in nums2 = [1,2,3,4]. There is no next greater element, so the answer is -1.
'''


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = []
        
        n = len(nums2)
        m = max(nums2)
        
        for num in nums1:
            done = False
            
            if num == m:
                res.append(-1)
            else:
                for i in range(nums2.index(num) + 1, n):
                    if nums2[i] > num:
                        res.append(nums2[i])
                        done = True
                        break

                if done == False:
                    res.append(-1)
        
        return res

```

## Longest Harmonious Subsequence

```py
'''
Longest Harmonious Subsequence
Easy

We define a harmonious array as an array where the difference between its maximum value and its minimum value is exactly 1.

Given an integer array nums, return the length of its longest harmonious subsequence among all its possible subsequences.

A subsequence of array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.

 

Example 1:

Input: nums = [1,3,2,2,5,2,3,7]
Output: 5
Explanation: The longest harmonious subsequence is [3,2,2,2,3].

Example 2:

Input: nums = [1,2,3,4]
Output: 2

Example 3:

Input: nums = [1,1,1,1]
Output: 0
'''


class Solution:
    def findLHS(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        
        dic = {}
        res = 0
        
        for n in nums:
            if n in dic:
                dic[n] +=1
            else:
                dic[n] = 1
                
        for key, value in dic.items():
            if key + 1 in dic:
                res = max(res, dic[key] + dic[key + 1])
                
        return res

```

## Add Two Numbers

```py
'''
Add Two Numbers
Medium

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

Example 1:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        
        digit1 = ""
        cur1 = l1
        
        while cur1 != None:
            digit1 += str(cur1.val)
            cur1 = cur1.next
            
        digit2 = ""
        cur2 = l2
        
        while cur2 != None:
            digit2 += str(cur2.val)
            cur2 = cur2.next
            
        digit1 = digit1[::-1]
        digit2 = digit2[::-1]
            
        digit1 = int(digit1)
        digit2 = int(digit2)
        
        res = str(digit1 + digit2)
        res = res[::-1]
        
        head = ListNode(int(res[0]))
        res = res[1:]
        cur = head
        
        for char in res:
            cur.next = ListNode(int(char))
            cur = cur.next
            
        return head

```

## Jewels and Stones

```py
'''
Jewels and Stones
Easy

You're given strings jewels representing the types of stones that are jewels, and stones representing the stones you have. Each character in stones is a type of stone you have. You want to know how many of the stones you have are also jewels.

Letters are case sensitive, so "a" is considered a different type of stone from "A".

 

Example 1:

Input: jewels = "aA", stones = "aAAbbbb"
Output: 3

Example 2:

Input: jewels = "z", stones = "ZZ"
Output: 0
'''


class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        hs = set(list(jewels))
        
        res = 0
        
        for char in stones:
            if char in hs:
                res += 1
                
        return res

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

## Consecutive Characters

```py
'''
Consecutive Characters
Easy

Given a string s, the power of the string is the maximum length of a non-empty substring that contains only one unique character.

Return the power of the string.

 

Example 1:

Input: s = "leetcode"
Output: 2
Explanation: The substring "ee" is of length 2 with the character 'e' only.

Example 2:

Input: s = "abbcccddddeeeeedcba"
Output: 5
Explanation: The substring "eeeee" is of length 5 with the character 'e' only.

Example 3:

Input: s = "triplepillooooow"
Output: 5

Example 4:

Input: s = "hooraaaaaaaaaaay"
Output: 11

Example 5:

Input: s = "tourist"
Output: 1
'''


class Solution:
    def maxPower(self, s: str) -> int:
        if len(s) == 1:
            return 1
        
        max_len = 1
        
        left = 0
        right = 0
        
        while right < len(s):
            cache = s[right]
            
            right += 1
            
            if right < len(s) and s[right] == cache:
                max_len = max(max_len, right - left + 1)
                continue
            else:
                left = right
                
        return max_len

```

## Sort Characters By Frequency

```py
'''
Sort Characters By Frequency
Medium

Given a string s, sort it in decreasing order based on the frequency of characters, and return the sorted string.

 

Example 1:

Input: s = "tree"
Output: "eert"
Explanation: 'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

Example 2:

Input: s = "cccaaa"
Output: "aaaccc"
Explanation: Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
Note that "cacaca" is incorrect, as the same characters must be together.

Example 3:

Input: s = "Aabb"
Output: "bbAa"
Explanation: "bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.
'''


class Solution:
    def frequencySort(self, s: str) -> str:
        dic = {}
        
        s = [char for char in s]
        
        for char in s:
            if char in dic:
                dic[char] += 1
            else:
                dic[char] = 1
        
        res = ""
        for key in sorted(dic, key=lambda x: dic[x], reverse=True):
            temp = key * dic[key]
            res += temp
            
        return res

```

## Base 7

```py
'''
Base 7
Easy

Given an integer num, return a string of its base 7 representation.

 

Example 1:

Input: num = 100
Output: "202"

Example 2:

Input: num = -7
Output: "-10"
'''


class Solution:
    def convertToBase7(self, num: int) -> str:
        if num == 0:
            return "0"
        
        res = ""
        flag = True
        
        if num < 0:
            flag = False
            num = abs(num)
                    
        while num > 0:
            res += str(num % 7)
            num = num // 7
        
        return res[::-1] if flag == True else f"-{res[::-1]}"

```

## 3Sum

```py
'''
3Sum
Medium

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Example 2:

Input: nums = []
Output: []

Example 3:

Input: nums = [0]
Output: []
'''



class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []
        
        nums.sort()
        
        res = []
        
        for i in range(len(nums)-2):
            if i == 0 or (i > 0 and nums[i] != nums[i-1]):                
                left, right = i+1, len(nums) - 1
                target = 0 - nums[i]
                
                while left < right:
                    if nums[left] + nums[right] == target:
                        res.append([nums[left], nums[right], nums[i]])
                        
                        while left < right and nums[left] == nums[left+1]:
                            left += 1
                        while left < right and nums[right] == nums[right-1]:
                            right -= 1
                            
                        left += 1
                        right -= 1
                        
                    elif nums[left] + nums[right] > target:
                        right -= 1
                    else:
                        left += 1
        
        return res

```

## Reshape the Matrix

```py
'''
Reshape the Matrix
Easy

In MATLAB, there is a handy function called reshape which can reshape an m x n matrix into a new one with a different size r x c keeping its original data.

You are given an m x n matrix mat and two integers r and c representing the number of rows and the number of columns of the wanted reshaped matrix.

The reshaped matrix should be filled with all the elements of the original matrix in the same row-traversing order as they were.

If the reshape operation with given parameters is possible and legal, output the new reshaped matrix; Otherwise, output the original matrix.

 

Example 1:

Input: mat = [[1,2],[3,4]], r = 1, c = 4
Output: [[1,2,3,4]]

Example 2:

Input: mat = [[1,2],[3,4]], r = 2, c = 4
Output: [[1,2],[3,4]]
'''


class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        vals = []
        
        for row in mat:
            vals.extend(row)
            
        if r * c != len(vals):
            return mat
        
        elif r == 1:
            return [vals]
        
        res = []
        
        for i in range(r):
            t = vals[c*i:(c*i)+c]
            res.append(t)
            
        return res

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

## Design HashSet

```py
'''
Design HashSet
Easy

Design a HashSet without using any built-in hash table libraries.

Implement MyHashSet class:

    void add(key) Inserts the value key into the HashSet.
    bool contains(key) Returns whether the value key exists in the HashSet or not.
    void remove(key) Removes the value key in the HashSet. If key does not exist in the HashSet, do nothing.

 

Example 1:

Input
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
Output
[null, null, null, true, false, null, true, null, false]

Explanation
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // return True
myHashSet.contains(3); // return False, (not found)
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // return True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // return False, (already removed)

 

Constraints:

    0 <= key <= 106
    At most 104 calls will be made to add, remove, and contains.
'''


class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.arr = [False] * ((10 ** 6) + 1)
        

    def add(self, key: int) -> None:
        self.arr[key] = True
        

    def remove(self, key: int) -> None:
        self.arr[key] = False
        

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        return self.arr[key]
        


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)

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

## Kth Smallest Element in a Sorted Matrix

```py
'''
Kth Smallest Element in a Sorted Matrix
Medium

Given an n x n matrix where each of the rows and columns are sorted in ascending order, return the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

 

Example 1:

Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13
Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13

Example 2:

Input: matrix = [[-5]], k = 1
Output: -5
'''

class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        def merge(arr1, arr2):
            l1 = 0
            l2 = 0
            
            res = []
            
            while l1 < len(arr1) and l2 < len(arr2):
                if arr1[l1] <= arr2[l2]:
                    res.append(arr1[l1])
                    l1 += 1
                else:
                    res.append(arr2[l2])
                    l2 += 1
                    
            while l1 < len(arr1):
                res.append(arr1[l1])
                l1 += 1
                
            while l2 < len(arr2):
                res.append(arr2[l2])
                l2 += 1
                    
            return res
        
        elements = []
        
        for arr in matrix:
            elements = merge(elements, arr)
            
        return elements[k-1]

```

## Minimum Path Sum

```py
'''
Minimum Path Sum
Medium

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

 

Example 1:

Input: grid = [
                [1,3,1],
                [1,5,1],
                [4,2,1]
              ]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.

Example 2:

Input: grid = [[1,2,3],[4,5,6]]
Output: 12
'''


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = []
        
        for i in range(len(grid)):
            dp.append([0] * len(grid[i]))
                    
        for i in range(len(dp)):
            for j in range(len(dp[i])):
                dp[i][j] += grid[i][j]
                
                if i > 0 and j > 0:
                    dp[i][j] += min(dp[i-1][j], dp[i][j-1])
                    
                elif i > 0:
                    dp[i][j] += dp[i-1][j]
                elif j > 0:
                    dp[i][j] += dp[i][j-1]
         
        return dp[-1][-1]

```

## Maximum Score After Splitting a String

```py
'''
Maximum Score After Splitting a String
Easy

Given a string s of zeros and ones, return the maximum score after splitting the string into two non-empty substrings (i.e. left substring and right substring).

The score after splitting a string is the number of zeros in the left substring plus the number of ones in the right substring.

 

Example 1:

Input: s = "011101"
Output: 5 
Explanation: 
All possible ways of splitting s into two non-empty substrings are:
left = "0" and right = "11101", score = 1 + 4 = 5 
left = "01" and right = "1101", score = 1 + 3 = 4 
left = "011" and right = "101", score = 1 + 2 = 3 
left = "0111" and right = "01", score = 1 + 1 = 2 
left = "01110" and right = "1", score = 2 + 1 = 3

Example 2:

Input: s = "00111"
Output: 5
Explanation: When left = "00" and right = "111", we get the maximum score = 2 + 3 = 5

Example 3:

Input: s = "1111"
Output: 3
'''


class Solution:
    def maxScore(self, s: str) -> int:
        i = 1
        
        max_val = -1
        
        left = s[:i]
        right = s[i:]
            
        c1 = left.count('0')
        c2 =  right.count('1')
                    
        max_val = max(max_val, c1 + c2)
            
        i += 1
        
        while i < len(s):
            left = s[:i]
            right = s[i:]
            
            c1 = left.count('0')
            c2 =  right.count('1')
                        
            max_val = max(max_val, c1 + c2)
            
            i += 1
            
        return max_val

```

## Design HashMap

```py
'''
Design HashMap
Easy

Design a HashMap without using any built-in hash table libraries.

Implement the MyHashMap class:

    MyHashMap() initializes the object with an empty map.
    void put(int key, int value) inserts a (key, value) pair into the HashMap. If the key already exists in the map, update the corresponding value.
    int get(int key) returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
    void remove(key) removes the key and its corresponding value if the map contains the mapping for the key.

 

Example 1:

Input
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
Output
[null, null, null, 1, -1, null, 1, null, -1]

Explanation
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // The map is now [[1,1]]
myHashMap.put(2, 2); // The map is now [[1,1], [2,2]]
myHashMap.get(1);    // return 1, The map is now [[1,1], [2,2]]
myHashMap.get(3);    // return -1 (i.e., not found), The map is now [[1,1], [2,2]]
myHashMap.put(2, 1); // The map is now [[1,1], [2,1]] (i.e., update the existing value)
myHashMap.get(2);    // return 1, The map is now [[1,1], [2,1]]
myHashMap.remove(2); // remove the mapping for 2, The map is now [[1,1]]
myHashMap.get(2);    // return -1 (i.e., not found), The map is now [[1,1]]

 

Constraints:

    0 <= key, value <= 106
    At most 104 calls will be made to put, get, and remove.
'''


class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.arr = [-1] * ((10 ** 6) + 1)
        

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        self.arr[key] = value
        

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        return self.arr[key]
        

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        self.arr[key] = -1
        


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

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

## Third Maximum Number

```py
'''
Third Maximum Number
Easy

Given integer array nums, return the third maximum number in this array. If the third maximum does not exist, return the maximum number.

 

Example 1:

Input: nums = [3,2,1]
Output: 1
Explanation: The third maximum is 1.

Example 2:

Input: nums = [1,2]
Output: 2
Explanation: The third maximum does not exist, so the maximum (2) is returned instead.

Example 3:

Input: nums = [2,2,3,1]
Output: 1
Explanation: Note that the third maximum here means the third maximum distinct number.
Both numbers with value 2 are both considered as second maximum.
'''


class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        s = set()
        for num in nums:
            s.add(num)
            
        if len(s) < 3:
            return max(s)
        
        for _ in range(2):
            s.remove(max(s))
        return max(s)

```

## Longest Continuous Increasing Subsequence

```py
'''
Longest Continuous Increasing Subsequence
Easy

Given an unsorted array of integers nums, return the length of the longest continuous increasing subsequence (i.e. subarray). The subsequence must be strictly increasing.

A continuous increasing subsequence is defined by two indices l and r (l < r) such that it is [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] and for each l <= i < r, nums[i] < nums[i + 1].

 

Example 1:

Input: nums = [1,3,5,4,7]
Output: 3
Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 7 are separated by element
4.

Example 2:

Input: nums = [2,2,2,2,2]
Output: 1
Explanation: The longest continuous increasing subsequence is [2] with length 1. Note that it must be strictly
increasing.
'''


class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 1
        
        max_len = 1
        cur_len = 1
        
        i = 1
        
        while i < len(nums):
            if nums[i-1] < nums[i]:
                cur_len += 1
            else:
                cur_len = 1
                
            max_len = max(max_len, cur_len)
            i += 1
            
        return max_len

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

