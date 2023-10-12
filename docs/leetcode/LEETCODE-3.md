
# leetcode programs:
## Page: 3
## Subtree of Another Tree

```py
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

## Count Special Quadruplets

```py
'''
Count Special Quadruplets
Easy

Given a 0-indexed integer array nums, return the number of distinct quadruplets (a, b, c, d) such that:

    nums[a] + nums[b] + nums[c] == nums[d], and
    a < b < c < d

 

Example 1:

Input: nums = [1,2,3,6]
Output: 1
Explanation: The only quadruplet that satisfies the requirement is (0, 1, 2, 3) because 1 + 2 + 3 == 6.

Example 2:

Input: nums = [3,3,6,4,5]
Output: 0
Explanation: There are no such quadruplets in [3,3,6,4,5].

Example 3:

Input: nums = [1,1,1,3,5]
Output: 4
Explanation: The 4 quadruplets that satisfy the requirement are:
- (0, 1, 2, 3): 1 + 1 + 1 == 3
- (0, 1, 3, 4): 1 + 1 + 3 == 5
- (0, 2, 3, 4): 1 + 1 + 3 == 5
- (1, 2, 3, 4): 1 + 1 + 3 == 5
'''


class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        count = 0
        
        n = len(nums)
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):     
                        if nums[i] + nums[j] + nums[k] == nums[l]:
                            count += 1
                            
        return count

```

## Find Numbers with Even Number of Digits

```py
'''
Find Numbers with Even Number of Digits
Easy
Given an array nums of integers, return how many of them contain an even number of digits.

 

Example 1:

Input: nums = [12,345,2,6,7896]
Output: 2
Explanation: 
12 contains 2 digits (even number of digits). 
345 contains 3 digits (odd number of digits). 
2 contains 1 digit (odd number of digits). 
6 contains 1 digit (odd number of digits). 
7896 contains 4 digits (even number of digits). 
Therefore only 12 and 7896 contain an even number of digits.

Example 2:

Input: nums = [555,901,482,1771]
Output: 1 
Explanation: 
Only 1771 contains an even number of digits.
'''


class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        
        for num in nums:
            if len(str(num)) % 2 == 0:
                count += 1
                
        return count

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

## Valid Perfect Square

```py
'''
Valid Perfect Square
Easy

Given a positive integer num, write a function which returns True if num is a perfect square else False.

Follow up: Do not use any built-in library function such as sqrt.

 

Example 1:

Input: num = 16
Output: true

Example 2:

Input: num = 14
Output: false
'''


class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num == 1:
            return True
        
        i = 2
        
        while (i * i) <= num:
            if (i * i ) == num:
                return True
            
            i += 1
            
        return False

```

## LRU Cache

```py
'''
LRU Cache
Medium

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

    LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    int get(int key) Return the value of the key if the key exists, otherwise return -1.
    void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.

 

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
'''



class LRUCache:
    # just have a count variable to denote when the cache was accessed. It acts like a time stamp. lower the count lower the priority

    def __init__(self, capacity: int):
        self.cache = {}
        self.count = 0
        self.cap = capacity

    def get(self, key: int) -> int:
        if key in self.cache:
            self.count += 1
            self.cache[key][0] = self.count
            return self.cache[key][1]
        else:
            return -1
            

    def put(self, key: int, value: int) -> None:
        if len(self.cache) == self.cap:
            if key not in self.cache:
                last_used_val = self.count
                last_used_key = -1
                for k, v in self.cache.items():
                    if last_used_val >= v[0]:
                        last_used_val = v[0]
                        last_used_key = k

                self.cache.pop(last_used_key)
                self.count += 1
                self.cache[key] = [self.count, value]
            else:
                self.count += 1
                self.cache[key] = [self.count, value]
                
        else:
            self.count += 1
            self.cache[key] = [self.count, value]         

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

```

## Maximum Nesting Depth of the Parentheses

```py
'''
Maximum Nesting Depth of the Parentheses
Easy

A string is a valid parentheses string (denoted VPS) if it meets one of the following:

    It is an empty string "", or a single character not equal to "(" or ")",
    It can be written as AB (A concatenated with B), where A and B are VPS's, or
    It can be written as (A), where A is a VPS.

We can similarly define the nesting depth depth(S) of any VPS S as follows:

    depth("") = 0
    depth(C) = 0, where C is a string with a single character not equal to "(" or ")".
    depth(A + B) = max(depth(A), depth(B)), where A and B are VPS's.
    depth("(" + A + ")") = 1 + depth(A), where A is a VPS.

For example, "", "()()", and "()(()())" are VPS's (with nesting depths 0, 1, and 2), and ")(" and "(()" are not VPS's.

Given a VPS represented as string s, return the nesting depth of s.

 

Example 1:

Input: s = "(1+(2*3)+((8)/4))+1"
Output: 3
Explanation: Digit 8 is inside of 3 nested parentheses in the string.

Example 2:

Input: s = "(1)+((2))+(((3)))"
Output: 3

Example 3:

Input: s = "1+(2*3)/(2-1)"
Output: 1

Example 4:

Input: s = "1"
Output: 0
'''


class Solution:
    def maxDepth(self, s: str) -> int:
        stack = []
        
        res = 0
        
        for char in s:
            if char == '(':
                stack.append(char)
                res = max(res, len(stack))
                
            if char == ')':
                stack.pop()
                
            res = max(res, len(stack))
            
        return res

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

## Factorial Trailing Zeroes

```py
'''
Factorial Trailing Zeroes
Easy

Given an integer n, return the number of trailing zeroes in n!.

Follow up: Could you write a solution that works in logarithmic time complexity?

 

Example 1:

Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.

Example 2:

Input: n = 5
Output: 1
Explanation: 5! = 120, one trailing zero.

Example 3:

Input: n = 0
Output: 0
'''


class Solution:
    def trailingZeroes(self, n: int) -> int:
        fact = 1
        
        while n > 0:
            fact *= n
            n -= 1
            
        fact = str(fact)[::-1]
        res = 0
        
        for char in fact:
            if char != '0':
                return res
            
            res += 1
        
        return res

```

## Find Mode in Binary Search Tree

```py
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

```

## Valid Boomerang

```py
'''
Valid Boomerang
Easy

Given an array points where points[i] = [xi, yi] represents a point on the X-Y plane, return true if these points are a boomerang.

A boomerang is a set of three points that are all distinct and not in a straight line.

 

Example 1:

Input: points = [[1,1],[2,3],[3,2]]
Output: true

Example 2:

Input: points = [[1,1],[2,2],[3,3]]
Output: false
'''


class Solution:
    def isBoomerang(self, points: List[List[int]]) -> bool:
        p1, p2, p3 = points
        
        return (p2[1] - p1[1]) * (p3[0] - p1[0]) != (p2[0] - p1[0]) * (p3[1] - p1[1])

```

## Hamming Distance

```py
'''
Hamming Distance
Easy

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers x and y, return the Hamming distance between them.

 

Example 1:

Input: x = 1, y = 4
Output: 2
Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
The above arrows point to positions where the corresponding bits are different.

Example 2:

Input: x = 3, y = 1
Output: 1
'''


class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        bits1 = bin(x)[2:]
        bits2 = bin(y)[2:]
        
        dist = 0
        
        if len(bits1) < len(bits2):
            temp = len(bits2) - len(bits1)
            
            bits1 = ("0" * temp) + bits1
            
        elif len(bits2) < len(bits1):
            temp = len(bits1) - len(bits2)
            
            bits2 = ("0" * temp) + bits2
        
        for b1, b2 in zip(list(bits1), list(bits2)):
            if b1 != b2:
                dist += 1
                
        return dist

```

## Coin Change

```py
'''
Coin Change
Medium

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:

Input: coins = [2], amount = 3
Output: -1

Example 3:

Input: coins = [1], amount = 0
Output: 0

Example 4:

Input: coins = [1], amount = 1
Output: 1

Example 5:

Input: coins = [1], amount = 2
Output: 2
'''


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        
        dp[0] = 0
        
        for a in range(1, amount + 1):
            for coin in coins:
                if a - coin >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - coin])
                    
        return dp[amount] if dp[amount] != (amount + 1) else -1

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

## Longest Palindrome

```py
'''
Longest Palindrome
Easy

Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters.

Letters are case sensitive, for example, "Aa" is not considered a palindrome here.

 

Example 1:

Input: s = "abccccdd"
Output: 7
Explanation:
One longest palindrome that can be built is "dccaccd", whose length is 7.

Example 2:

Input: s = "a"
Output: 1

Example 3:

Input: s = "bb"
Output: 2
'''


class Solution:
    def longestPalindrome(self, s: str) -> int:        
        if len(s) == 1:
            return 1
        
        dic = {}
    
        for char in s:
            if char in dic:
                dic[char] += 1
            else:
                dic[char] = 1
                
        max_len = 0
        
        for val in dic.values():
            max_len += (val // 2) * 2
            
            if max_len % 2 == 0 and val % 2 == 1:
                max_len += 1
                
        return max_len

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

## Teemo Attacking

```py
'''
Teemo Attacking
Easy

Our hero Teemo is attacking an enemy Ashe with poison attacks! When Teemo attacks Ashe, Ashe gets poisoned for a exactly duration seconds. More formally, an attack at second t will mean Ashe is poisoned during the inclusive time interval [t, t + duration - 1]. If Teemo attacks again before the poison effect ends, the timer for it is reset, and the poison effect will end duration seconds after the new attack.

You are given a non-decreasing integer array timeSeries, where timeSeries[i] denotes that Teemo attacks Ashe at second timeSeries[i], and an integer duration.

Return the total number of seconds that Ashe is poisoned.

 

Example 1:

Input: timeSeries = [1,4], duration = 2
Output: 4
Explanation: Teemo's attacks on Ashe go as follows:
- At second 1, Teemo attacks, and Ashe is poisoned for seconds 1 and 2.
- At second 4, Teemo attacks, and Ashe is poisoned for seconds 4 and 5.
Ashe is poisoned for seconds 1, 2, 4, and 5, which is 4 seconds in total.

Example 2:

Input: timeSeries = [1,2], duration = 2
Output: 3
Explanation: Teemo's attacks on Ashe go as follows:
- At second 1, Teemo attacks, and Ashe is poisoned for seconds 1 and 2.
- At second 2 however, Teemo attacks again and resets the poison timer. Ashe is poisoned for seconds 2 and 3.
Ashe is poisoned for seconds 1, 2, and 3, which is 3 seconds in total
'''


class Solution:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        seconds = 0
        
        if len(timeSeries) == 1:
            return duration
        
        for i in range(len(timeSeries)-1):
            seconds += min(duration, timeSeries[i+1] - timeSeries[i])
            
        return seconds + duration

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

## Keyboard Row

```py
'''
Keyboard Row
Easy

Given an array of strings words, return the words that can be typed using letters of the alphabet on only one row of American keyboard like the image below.

In the American keyboard:

    the first row consists of the characters "qwertyuiop",
    the second row consists of the characters "asdfghjkl", and
    the third row consists of the characters "zxcvbnm".

 

Example 1:

Input: words = ["Hello","Alaska","Dad","Peace"]
Output: ["Alaska","Dad"]

Example 2:

Input: words = ["omk"]
Output: []

Example 3:

Input: words = ["adsdf","sfd"]
Output: ["adsdf","sfd"]
'''


class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        
        row1 = "qwertyuiop"
        row2 = "asdfghjkl"
        row3 = "zxcvbnm"
        
        res = []
        
        for word in words:
            if word[0].lower() in row1:
                print(f"{word} in row1")
                for char in word:
                    if char.lower() not in row1:
                        break
                        
                else:
                    res.append(word)
                
            elif word[0].lower() in row2:
                print(f"{word} in row2")
                for char in word:
                    if char.lower() not in row2:
                        break
                        
                else:
                    res.append(word)
                    
            elif word[0].lower() in row3:
                print(f"{word} in row3")
                for char in word:
                    if char.lower() not in row3:
                        break
                        
                else:
                    res.append(word)
                    
        return res

```

## Partition List

```py
'''
Partition List
Medium

Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

 

Example 1:

Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Example 2:

Input: head = [2,1], x = 2
Output: [1,2]
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        if head == None:
            return head
        
        arr = []
        cur = head
        
        while cur != None:
            arr.append(cur.val)
            cur = cur.next
            
        left = []
        right = []
        
        for a in arr:
            if a < x:
                left.append(a)
            else:
                right.append(a)
                
        arr = left + right
        cur = head
        
        while cur != None:
            cur.val = arr[0]
            cur = cur.next
            arr.pop(0)
            
        return head

```

