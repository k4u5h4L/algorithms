
# leetcode programs:
## Page: 4
## House Robber

```py
'''
House Robber
Medium

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
'''


class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return max(nums)
        
        dp = [0] * len(nums) 
        
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
            
        return dp[-1]

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

## Kth Largest Element in a Stream

```py
'''
Kth Largest Element in a Stream
Easy

Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Implement KthLargest class:

    KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of integers nums.
    int add(int val) Appends the integer val to the stream and returns the element representing the kth largest element in the stream.

 

Example 1:

Input
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
Output
[null, 4, 5, 5, 8, 8]

Explanation
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8
'''


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        nums.sort(reverse=True)
        self.nums = nums
        

    def add(self, val: int) -> int:
        self.nums.append(val)
        self.nums.sort(reverse=True)
                
        return self.nums[self.k-1]
        


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)

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

## Integer Replacement

```py
'''
Integer Replacement
Medium

Given a positive integer n, you can apply one of the following operations:

    If n is even, replace n with n / 2.
    If n is odd, replace n with either n + 1 or n - 1.

Return the minimum number of operations needed for n to become 1.

 

Example 1:

Input: n = 8
Output: 3
Explanation: 8 -> 4 -> 2 -> 1

Example 2:

Input: n = 7
Output: 4
Explanation: 7 -> 8 -> 4 -> 2 -> 1
or 7 -> 6 -> 3 -> 2 -> 1

Example 3:

Input: n = 4
Output: 2
'''


class Solution:
    def integerReplacement(self, n: int) -> int:
        return self.num_steps(n) - 1
    
    def num_steps(self, n, memo={}):
        if n in memo:
            return memo[n]
        
        if n == 1:
            return 1
        
        elif n % 2 == 0:
            memo[n] = 1 + self.num_steps(n // 2, memo)
            
        else:
            memo[n] = 1 + min(self.num_steps(n + 1, memo), self.num_steps(n - 1, memo))
        
        return memo[n]

```

## Number of Segments in a String

```py
'''
Number of Segments in a String
Easy

You are given a string s, return the number of segments in the string. 

A segment is defined to be a contiguous sequence of non-space characters.

 

Example 1:

Input: s = "Hello, my name is John"
Output: 5
Explanation: The five segments are ["Hello,", "my", "name", "is", "John"]

Example 2:

Input: s = "Hello"
Output: 1

Example 3:

Input: s = "love live! mu'sic forever"
Output: 4

Example 4:

Input: s = ""
Output: 0
'''


class Solution:
    def countSegments(self, s: str) -> int:
        if s == "":
            return 0
        
        return len([char for char in s.split(" ") if char != ""])

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

## Increasing Decreasing String

```py
'''
Increasing Decreasing String
Easy

Given a string s. You should re-order the string using the following algorithm:

    Pick the smallest character from s and append it to the result.
    Pick the smallest character from s which is greater than the last appended character to the result and append it.
    Repeat step 2 until you cannot pick more characters.
    Pick the largest character from s and append it to the result.
    Pick the largest character from s which is smaller than the last appended character to the result and append it.
    Repeat step 5 until you cannot pick more characters.
    Repeat the steps from 1 to 6 until you pick all characters from s.

In each step, If the smallest or the largest character appears more than once you can choose any occurrence and append it to the result.

Return the result string after sorting s with this algorithm.

 

Example 1:

Input: s = "aaaabbbbcccc"
Output: "abccbaabccba"
Explanation: After steps 1, 2 and 3 of the first iteration, result = "abc"
After steps 4, 5 and 6 of the first iteration, result = "abccba"
First iteration is done. Now s = "aabbcc" and we go back to step 1
After steps 1, 2 and 3 of the second iteration, result = "abccbaabc"
After steps 4, 5 and 6 of the second iteration, result = "abccbaabccba"

Example 2:

Input: s = "rat"
Output: "art"
Explanation: The word "rat" becomes "art" after re-ordering it with the mentioned algorithm.

Example 3:

Input: s = "leetcode"
Output: "cdelotee"

Example 4:

Input: s = "ggggggg"
Output: "ggggggg"

Example 5:

Input: s = "spo"
Output: "ops"
'''


class Solution:
    def sortString(self, s: str) -> str:
        dic = {
            'a' : 0, 'b' : 0, 'c' : 0, 'd' : 0, 'e' : 0, 'f' : 0,'g' : 0, 'h' : 0, 'i' : 0, 'j' : 0, 'k' : 0, 'l' : 0,'m' : 0, 'n' : 0, 'o' : 0, 'p' : 0, 'q' : 0, 'r' : 0,'s' : 0, 't' : 0, 'u' : 0, 'v' : 0, 'w' : 0, 'x' : 0,'y' : 0, 'z' : 0 
        }
        
        for c in s:
            dic[c] += 1
            
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        res = ""
        
        while len(res) < len(s):
            for c in alphabet:
                if dic[c] > 0:
                    res += c
                    dic[c] -= 1
                    
            for c in reversed(alphabet):
                if dic[c] > 0:
                    res += c
                    dic[c] -= 1
                    
        return res

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

## Pow(x,n)

```py
'''
Pow(x, n)
Medium

Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).

 

Example 1:

Input: x = 2.00000, n = 10
Output: 1024.00000

Example 2:

Input: x = 2.10000, n = 3
Output: 9.26100

Example 3:

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/2^2 = 1/4 = 0.25
'''

class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == float(1) or n == 0:
            return 1
        
        if x == float(-1):
            return 1 if n % 2 == 0 else -1
        
        neg = False
        
        if n < 0:
            n = abs(n)
            neg = True
            
        if neg == True:
            x = 1 / x
            
        res = x
        for _ in range(n-1):
            if abs(res) < 0.00001:
                return 0.00000
            elif abs(res) > 100000:
                return 100000
            res *= x
            
        return res

```

## Perfect Number

```py
'''
Perfect Number
Easy

A perfect number is a positive integer that is equal to the sum of its positive divisors, excluding the number itself. A divisor of an integer x is an integer that can divide x evenly.

Given an integer n, return true if n is a perfect number, otherwise return false.

 

Example 1:

Input: num = 28
Output: true
Explanation: 28 = 1 + 2 + 4 + 7 + 14
1, 2, 4, 7, and 14 are all divisors of 28.

Example 2:

Input: num = 6
Output: true

Example 3:

Input: num = 496
Output: true

Example 4:

Input: num = 8128
Output: true

Example 5:

Input: num = 2
Output: false
'''

# [accepted] optimal solution

class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num <= 0:
            return False
        
        div_sum = 0
        i = 1
        
        while (i * i) <= num:
            if num % i == 0:
                div_sum += i
                
                if (i * i) != num:
                    div_sum += num // i
            
            i += 1
            
        return div_sum - num == num


# [accepted] very simple solution, but need to know what you're doing and not exactly an "algorithm" 

class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        return num in (6, 28, 496, 8128, 33550336)
      
      
 # [time limit exceeded] brute force solution, may slow down for large numbers

class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        divisors_sum = 0
        
        for i in range(1, num):
            if num % i == 0:
                divisors_sum += i
                
        return divisors_sum == num

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

## Find Center of Star Graph

```py
'''
Find Center of Star Graph
Easy

There is an undirected star graph consisting of n nodes labeled from 1 to n. A star graph is a graph where there is one center node and exactly n - 1 edges that connect the center node with every other node.

You are given a 2D integer array edges where each edges[i] = [ui, vi] indicates that there is an edge between the nodes ui and vi. Return the center of the given star graph.

 

Example 1:

Input: edges = [[1,2],[2,3],[4,2]]
Output: 2
Explanation: As shown in the figure above, node 2 is connected to every other node, so 2 is the center.

Example 2:

Input: edges = [[1,2],[5,1],[1,3],[1,4]]
Output: 1
'''


class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        e = edges[0]
        edges = edges[1:]
        
        if e[0] in edges[0]:
            return e[0]
        else:
            return e[1]

```

## Find the Distance Value Between Two Arrays

```py
'''
Find the Distance Value Between Two Arrays
Easy

Given two integer arrays arr1 and arr2, and the integer d, return the distance value between the two arrays.

The distance value is defined as the number of elements arr1[i] such that there is not any element arr2[j] where |arr1[i]-arr2[j]| <= d.

 

Example 1:

Input: arr1 = [4,5,8], arr2 = [10,9,1,8], d = 2
Output: 2
Explanation: 
For arr1[0]=4 we have: 
|4-10|=6 > d=2 
|4-9|=5 > d=2 
|4-1|=3 > d=2 
|4-8|=4 > d=2 
For arr1[1]=5 we have: 
|5-10|=5 > d=2 
|5-9|=4 > d=2 
|5-1|=4 > d=2 
|5-8|=3 > d=2
For arr1[2]=8 we have:
|8-10|=2 <= d=2
|8-9|=1 <= d=2
|8-1|=7 > d=2
|8-8|=0 <= d=2

Example 2:

Input: arr1 = [1,4,2,3], arr2 = [-4,-3,6,10,20,30], d = 3
Output: 2

Example 3:

Input: arr1 = [2,1,100,3], arr2 = [-5,-2,10,-3,7], d = 6
Output: 1
'''


class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        count = 0
        
        for a1 in arr1:
            flag = True
            
            for a2 in arr2:
                if abs(a1 - a2) <= d:
                    flag = False
                    break
            
            if flag == True:
                count += 1
        
        return count

```

## Set Mismatch

```py
'''
Set Mismatch
Easy

You have a set of integers s, which originally contains all the numbers from 1 to n. Unfortunately, due to some error, one of the numbers in s got duplicated to another number in the set, which results in repetition of one number and loss of another number.

You are given an integer array nums representing the data status of this set after the error.

Find the number that occurs twice and the number that is missing and return them in the form of an array.

 

Example 1:

Input: nums = [1,2,2,4]
Output: [2,3]

Example 2:

Input: nums = [1,1]
Output: [1,2]
'''


class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:        
        hs = set()
        res = [-1, -1]
        
        for n in nums:
            if n in hs:
                res[0] = n
            else:
                hs.add(n)
                
        for i in range(1, len(nums)+1):
            if i not in hs:
                res[1] = i
                break
                
        return res

```

## Find Lucky Integer in an Array

```py
'''
Find Lucky Integer in an Array
Easy

Given an array of integers arr, a lucky integer is an integer which has a frequency in the array equal to its value.

Return a lucky integer in the array. If there are multiple lucky integers return the largest of them. If there is no lucky integer return -1.

 

Example 1:

Input: arr = [2,2,3,4]
Output: 2
Explanation: The only lucky number in the array is 2 because frequency[2] == 2.

Example 2:

Input: arr = [1,2,2,3,3,3]
Output: 3
Explanation: 1, 2 and 3 are all lucky numbers, return the largest of them.

Example 3:

Input: arr = [2,2,2,3,3]
Output: -1
Explanation: There are no lucky numbers in the array.

Example 4:

Input: arr = [5]
Output: -1

Example 5:

Input: arr = [7,7,7,7,7,7,7]
Output: 7
'''



class Solution:
    def findLucky(self, arr: List[int]) -> int:
        dic = {}
        
        for n in arr:
            if n in dic:
                dic[n] += 1
            else:
                dic[n] = 1
            
        res = -1
                
        for key, value in dic.items():
            if key == value:
                res = max(res, key)
                
        return res

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

## Longest Palindromic Substring

```py
'''
Longest Palindromic Substring
Medium

Given a string s, return the longest palindromic substring in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:

Input: s = "cbbd"
Output: "bb"

Example 3:

Input: s = "a"
Output: "a"

Example 4:

Input: s = "ac"
Output: "a"
'''

# pretty fast solution using sliding window

class Solution:
    def longestPalindrome(self, s: str) -> str:
        def is_palin(s):
            return s == s[::-1]
        
        if len(s) == 1:
            return s
        elif len(s) == 2:
            if s[0] == s[1]:
                return s
            else:
                return s[0]
        
        res = s[0]
        
        left = 0
        right = 1
        
        while right < len(s):
            temp = s[left:right+1]
            
            if is_palin(temp):
                right += 1
                
                if left > 0:
                    left -= 1
                
                if len(res) < len(temp):
                    res = temp
                    
                continue
                    
            if left >= right:
                right += 1
            else:
                left += 1
                    
        return res


# naive way, don't use this lol

class Solution:
    def longestPalindrome(self, s: str) -> str:
        def is_palin(s):
            return s == s[::-1]
        
        if len(s) == 1:
            return s
        elif len(s) == 2:
            if s[0] == s[1]:
                return s
            else:
                return s[0]
        
        res = s[0]
        
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                temp = s[i:j+1]
                
                if len(temp) > len(res) and is_palin(temp):
                    res = s[i:j+1]
                    
        return res
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

## Flatten Binary Tree to Linked List

```py
'''
Flatten Binary Tree to Linked List
Medium

Given the root of a binary tree, flatten the tree into a "linked list":

    The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
    The "linked list" should be in the same order as a pre-order traversal of the binary tree.

 

Example 1:

Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

Example 2:

Input: root = []
Output: []

Example 3:

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
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            return
        
        arr = []
        
        self.preorder(root, arr)
                
        for i in range(len(arr) - 1):
            root.val = arr[i]
            root.left = None
            
            if root.right == None:
                root.right = TreeNode()
                
            root = root.right
            
        if len(arr) >= 1:
            root.val = arr[-1]
        
        return
    
    def preorder(self, root, arr):
        if root == None:
            return
        arr.append(root.val)
        
        self.preorder(root.left, arr)
        self.preorder(root.right, arr)

```

## N-Repeated Element in Size 2N Array

```py
'''
N-Repeated Element in Size 2N Array
Easy

You are given an integer array nums with the following properties:

    nums.length == 2 * n.
    nums contains n + 1 unique elements.
    Exactly one element of nums is repeated n times.

Return the element that is repeated n times.

 

Example 1:

Input: nums = [1,2,3,3]
Output: 3

Example 2:

Input: nums = [2,1,2,5,3,2]
Output: 2

Example 3:

Input: nums = [5,1,5,2,5,3,5,4]
Output: 5
'''


class Solution:
    def repeatedNTimes(self, nums: List[int]) -> int:
        n = len(nums) // 2
        
        dic = {}
        
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
                
        for key, val in dic.items():
            if val == n:
                return key
            
        return -1

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

