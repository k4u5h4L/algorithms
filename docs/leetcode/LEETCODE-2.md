
# leetcode programs:
## Page: 2
## Intersection of Two Arrays

```py
'''
Intersection of Two Arrays
Easy

Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must be unique and you may return the result in any order.

 

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]

Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [9,4]
Explanation: [4,9] is also accepted.
'''


class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        s = set()
        
        for num in nums1:
            s.add(num)
            
        res = []
        
        for num in nums2:
            if num in s:
                res.append(num)
                s.remove(num)
                
        return res

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

## Check If All 1's Are at Least Length K Places Away

```py
'''
Check If All 1's Are at Least Length K Places Away
Easy

Given an array nums of 0s and 1s and an integer k, return True if all 1's are at least k places away from each other, otherwise return False.

 

Example 1:

Input: nums = [1,0,0,0,1,0,0,1], k = 2
Output: true
Explanation: Each of the 1s are at least 2 places away from each other.

Example 2:

Input: nums = [1,0,0,1,0,1], k = 2
Output: false
Explanation: The second 1 and third 1 are only one apart from each other.

Example 3:

Input: nums = [1,1,1,1,1], k = 0
Output: true

Example 4:

Input: nums = [0,1,0,1], k = 1
Output: true
'''


class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        spaces = []
        
        for i in range(len(nums)):
            if nums[i] == 1:
                spaces.append(i)
                                
        for i in range(len(spaces)-1):
            if abs(spaces[i+1] - spaces[i]) <= k:
                return False
            
        return True

```

## Guess Number Higher or Lower

```py
'''
Guess Number Higher or Lower
Easy

We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API int guess(int num), which returns 3 possible results:

    -1: The number I picked is lower than your guess (i.e. pick < num).
    1: The number I picked is higher than your guess (i.e. pick > num).
    0: The number I picked is equal to your guess (i.e. pick == num).

Return the number that I picked.

 

Example 1:

Input: n = 10, pick = 6
Output: 6

Example 2:

Input: n = 1, pick = 1
Output: 1

Example 3:

Input: n = 2, pick = 1
Output: 1

Example 4:

Input: n = 2, pick = 2
Output: 2
'''

# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:

class Solution:
    def guessNumber(self, n: int) -> int:
        if n == 1:
            return 1
                
        left = 1
        right = n
        
        res = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            res = guess(mid)
            
            if res == 0:
                return mid
            elif res == -1:
                right = mid - 1
            elif res == 1:
                left = mid + 1
                
        return -1

```

## Sum of Left Leaves

```py
'''
Sum of Left Leaves
Easy

Given the root of a binary tree, return the sum of all left leaves.

 

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: 24
Explanation: There are two left leaves in the binary tree, with values 9 and 15 respectively.

Example 2:

Input: root = [1]
Output: 0
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        res = [0]
        
        self.inorder(root, res)
        
        return res[0]
    
    def inorder(self, root, res):
        if root == None:
            return
        
        if root.left and not root.left.left and not root.left.right:
            res[0] += root.left.val
            
        self.inorder(root.left, res)
        self.inorder(root.right, res)

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

## Majority Element II

```py
'''
Majority Element II
Medium

Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

Follow-up: Could you solve the problem in linear time and in O(1) space?

 

Example 1:

Input: nums = [3,2,3]
Output: [3]

Example 2:

Input: nums = [1]
Output: [1]

Example 3:

Input: nums = [1,2]
Output: [1,2]
'''

class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        dic = {}
        
        times = len(nums) // 3
        
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
                
        res = []
        
        for num in nums:
            if num in dic and dic[num] > times:
                res.append(num)
                dic.pop(num)
                
        return res

```

## Intersection of Two Arrays II

```py
'''
Intersection of Two Arrays II
Easy

Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.

 

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]

Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Explanation: [9,4] is also accepted.
'''


class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dic1 = {}
        dic2 = {}
        
        for num in nums1:
            if num in dic1:
                dic1[num] += 1
            else:
                dic1[num] = 1
                
        for num in nums2:
            if num in dic2:
                dic2[num] += 1
            else:
                dic2[num] = 1
                
        dic = {}
        
        for x in dic1:
            if x in dic2:
                dic[x] = min(dic1[x], dic2[x])
        
        res = []
        
        for key in dic.keys():
            res.extend([key] * dic[key])
            
        return res

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

## Most Common Word

```py
'''
Most Common Word
Easy

Given a string paragraph and a string array of the banned words banned, return the most frequent word that is not banned. It is guaranteed there is at least one word that is not banned, and that the answer is unique.

The words in paragraph are case-insensitive and the answer should be returned in lowercase.

 

Example 1:

Input: paragraph = "Bob hit a ball, the hit BALL flew far after it was hit.", banned = ["hit"]
Output: "ball"
Explanation: 
"hit" occurs 3 times, but it is a banned word.
"ball" occurs twice (and no other word does), so it is the most frequent non-banned word in the paragraph. 
Note that words in the paragraph are not case sensitive,
that punctuation is ignored (even if adjacent to words, such as "ball,"), 
and that "hit" isn't the answer even though it occurs more because it is banned.

Example 2:

Input: paragraph = "a.", banned = []
Output: "a"
'''


import string

class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        hs = set()
        
        paragraph = re.sub(f"[{string.punctuation}]", " ", paragraph)
        
        for word in banned:
            hs.add(word)
            
        paragraph = [word for word in paragraph.lower().split(" ") if word != ""]
        dic = {}
        
        for word in paragraph:
            if word not in hs:
                if word in dic:
                    dic[word] += 1
                else:
                    dic[word] = 1
                
        max_val = 0
        res = ""
        
        for key, value in dic.items():
            if value >=max_val:
                res = key
                max_val = value
                
        return res

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

## Truncate Sentence

```py
'''
Truncate Sentence
Easy

A sentence is a list of words that are separated by a single space with no leading or trailing spaces. Each of the words consists of only uppercase and lowercase English letters (no punctuation).

    For example, "Hello World", "HELLO", and "hello world hello world" are all sentences.

You are given a sentence s​​​​​​ and an integer k​​​​​​. You want to truncate s​​​​​​ such that it contains only the first k​​​​​​ words. Return s​​​​​​ after truncating it.

 

Example 1:

Input: s = "Hello how are you Contestant", k = 4
Output: "Hello how are you"
Explanation:
The words in s are ["Hello", "how" "are", "you", "Contestant"].
The first 4 words are ["Hello", "how", "are", "you"].
Hence, you should return "Hello how are you".

Example 2:

Input: s = "What is the solution to this problem", k = 4
Output: "What is the solution"
Explanation:
The words in s are ["What", "is" "the", "solution", "to", "this", "problem"].
The first 4 words are ["What", "is", "the", "solution"].
Hence, you should return "What is the solution".

Example 3:

Input: s = "chopper is not a tanuki", k = 5
Output: "chopper is not a tanuki"
'''

class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        words = s.split(" ")
        
        return ' '.join(words[:k])

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

## Kids With the Greatest Number of Candies

```py
'''
Kids With the Greatest Number of Candies
Easy

There are n kids with candies. You are given an integer array candies, where each candies[i] represents the number of candies the ith kid has, and an integer extraCandies, denoting the number of extra candies that you have.

Return a boolean array result of length n, where result[i] is true if, after giving the ith kid all the extraCandies, they will have the greatest number of candies among all the kids, or false otherwise.

Note that multiple kids can have the greatest number of candies.

 

Example 1:

Input: candies = [2,3,5,1,3], extraCandies = 3
Output: [true,true,true,false,true] 
Explanation: If you give all extraCandies to:
- Kid 1, they will have 2 + 3 = 5 candies, which is the greatest among the kids.
- Kid 2, they will have 3 + 3 = 6 candies, which is the greatest among the kids.
- Kid 3, they will have 5 + 3 = 8 candies, which is the greatest among the kids.
- Kid 4, they will have 1 + 3 = 4 candies, which is not the greatest among the kids.
- Kid 5, they will have 3 + 3 = 6 candies, which is the greatest among the kids.

Example 2:

Input: candies = [4,2,1,1,2], extraCandies = 1
Output: [true,false,false,false,false] 
Explanation: There is only 1 extra candy.
Kid 1 will always have the greatest number of candies, even if a different kid is given the extra candy.

Example 3:

Input: candies = [12,1,12], extraCandies = 10
Output: [true,false,true]
'''


class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        res = [False] * len(candies)
        
        for i in range(len(candies)):
            if (candies[i] + extraCandies) >= max(candies):
                res[i] = True
                
        return res

```

## Top K Frequent Elements

```py
'''
Top K Frequent Elements
Medium

Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:

Input: nums = [1], k = 1
Output: [1]
'''


from queue import PriorityQueue

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        def sort_func(x):
            return dic[x]
        dic = {}
        
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
                
        nums.sort(key=sort_func, reverse=True)
        
        s = set()
        res = []
        
        for num in nums:
            if len(s) == k:
                break
            if num in s:
                continue
            else:
                res.append(num)
                s.add(num)
        
        return res

```

## Build an Array With Stack Operations

```py
'''
Build an Array With Stack Operations
Easy

Given an array target and an integer n. In each iteration, you will read a number from  list = {1,2,3..., n}.

Build the target array using the following operations:

    Push: Read a new element from the beginning list, and push it in the array.
    Pop: delete the last element of the array.
    If the target array is already built, stop reading more elements.

Return the operations to build the target array. You are guaranteed that the answer is unique.

 

Example 1:

Input: target = [1,3], n = 3
Output: ["Push","Push","Pop","Push"]
Explanation: 
Read number 1 and automatically push in the array -> [1]
Read number 2 and automatically push in the array then Pop it -> [1]
Read number 3 and automatically push in the array -> [1,3]

Example 2:

Input: target = [1,2,3], n = 3
Output: ["Push","Push","Push"]

Example 3:

Input: target = [1,2], n = 4
Output: ["Push","Push"]
Explanation: You only need to read the first 2 numbers and stop.

Example 4:

Input: target = [2,3,4], n = 4
Output: ["Push","Pop","Push","Push","Push"]
'''


class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        hs = set(target)
        res = []
        count = 0
        
        for i in range(1, n+1):
            res.append("Push")
            count += 1
            if i not in hs:
                res.append("Pop")
                count -= 1
                
            if count >= len(target):
                break
                
        return res

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

## Minimum Cost to Connect Sticks

```py
'''
Minimim Cost to Connect Sticks
Medium

You have some sticks with positive integer lengths

You can connect any two sticks of lengths X andc Y by paying a cost X + Y.
You can perform this action until there is one stick remaining.

Return the minimum cost of connecting all the given sticks into one stick in this way.

Example 1:
Input: sticks = [2,4,3]
Output: 14

Example 2:
Input: sticks = [1,8,3,5]
Output: 30
'''

from queue import PriorityQueue

class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        cost = 0
        q = PriorityQueue()
        
        for stick in sticks:
            q.put(stick)
        
        while q.qsize() > 1:
            cur_sum = q.get() + q.get()
            cost += cur_sum
            q.put(cur_sum)
            
        return cost

```

## Insertion Sort List

```py
'''
Insertion Sort List
Medium

Given the head of a singly linked list, sort the list using insertion sort, and return the sorted list's head.

The steps of the insertion sort algorithm:

    Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
    At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
    It repeats until no input elements remain.

The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.

 

Example 1:

Input: head = [4,2,1,3]
Output: [1,2,3,4]

Example 2:

Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # if no elememt present
        if head == None:
            return head
        
        res = None
        
        cur = head
        
        while cur != None:
            # if result is empty, add one node and continue
            if res == None:
                res = ListNode(cur.val)
                cur = cur.next
                continue
                
            temp = ListNode(cur.val)
            cur1 = res
            prev = cur1
            
            # if only one element present, check where to insert it (back or front) and then continue
            if res.next == None:
                if res.val > temp.val:
                    temp.next = res
                    res = temp
                else:
                    res.next = temp
                    
                cur = cur.next
                continue
                
            # print(res)
            
            # if value is less than head, then no need to traverse list. just append at head and continue
            if temp.val < res.val:
                temp.next = res
                res = temp
                
                cur = cur.next
                continue
            
            # for any other case, traverse list and see the optimal place to insert new node
            while cur1 and temp.val >= cur1.val:
                prev = cur1
                cur1 = cur1.next
            
            prev.next = temp
            temp.next = cur1
            
            cur = cur.next
            
        return res

```

## Find Common Characters

```py
'''
Find Common Characters
Easy

Given a string array words, return an array of all characters that show up in all strings within the words (including duplicates). You may return the answer in any order.

 

Example 1:

Input: words = ["bella","label","roller"]
Output: ["e","l","l"]

Example 2:

Input: words = ["cool","lock","cook"]
Output: ["c","o"]
'''


class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        if len(words) == 1:
            return [char for char in words[0]]
        
        dic = {}
        
        for char in words[0]:
            if char in dic:
                dic[char] += 1
            else:
                dic[char] = 1
                
        words = words[1:]
        
        for word in words:
            temp_dic = dic.copy()
            dic = {}
            for char in word:
                if char in temp_dic and temp_dic[char] > 0:
                    temp_dic[char] -= 1
                    if char in dic:
                        dic[char] += 1
                    else:
                        dic[char] = 1
                   
        res = []
        for key, value in dic.items():
            res.extend([key] * value)
            
        return res

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

