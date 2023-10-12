
# leetcode programs:
## Page: 5
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

## Check if Binary String Has at Most One Segment of Ones

```py
'''
Check if Binary String Has at Most One Segment of Ones
Easy

Given a binary string s without leading zeros, return true if s contains at most one contiguous segment of ones. Otherwise, return false.

 

Example 1:

Input: s = "1001"
Output: false
Explanation: The ones do not form a contiguous segment.

Example 2:

Input: s = "110"
Output: true
'''

class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        s = [char for char in s.split('0') if char != '']
    
        if len(s) > 1:
            return False
        else:
            return True

```

## Longer Contiguous Segments of Ones than Zeros

```py
'''
Longer Contiguous Segments of Ones than Zeros
Easy

Given a binary string s, return true if the longest contiguous segment of 1s is strictly longer than the longest contiguous segment of 0s in s. Return false otherwise.

    For example, in s = "110100010" the longest contiguous segment of 1s has length 2, and the longest contiguous segment of 0s has length 3.

Note that if there are no 0s, then the longest contiguous segment of 0s is considered to have length 0. The same applies if there are no 1s.

 

Example 1:

Input: s = "1101"
Output: true
Explanation:
The longest contiguous segment of 1s has length 2: "1101"
The longest contiguous segment of 0s has length 1: "1101"
The segment of 1s is longer, so return true.

Example 2:

Input: s = "111000"
Output: false
Explanation:
The longest contiguous segment of 1s has length 3: "111000"
The longest contiguous segment of 0s has length 3: "111000"
The segment of 1s is not longer, so return false.

Example 3:

Input: s = "110100010"
Output: false
Explanation:
The longest contiguous segment of 1s has length 2: "110100010"
The longest contiguous segment of 0s has length 3: "110100010"
The segment of 1s is not longer, so return false.
'''


class Solution:
    def checkZeroOnes(self, s: str) -> bool:
        nums = [int(char) for char in s]
                        
        if len(nums) == 1:
            return True if nums[0] == 1 else False
        
        elif 1 not in nums:
            return False
        
        elif 0 not in nums:
            return True
        
        max_len1 = 1
        cur_len1 = 0
        
        max_len0 = 1
        cur_len0 = 0
        
        for val in nums:
            if val == 1:
                cur_len1 += 1
                cur_len0 = 0
            else:
                cur_len1 = 0
                cur_len0 += 1
                
            max_len1 = max(max_len1, cur_len1)
            max_len0 = max(max_len0, cur_len0)
            
        return True if max_len1 > max_len0 else False

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

## Binary Tree Right Side View

```py
'''
Binary Tree Right Side View
Medium

Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

 

Example 1:

Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

Example 2:

Input: root = [1,null,3]
Output: [1,3]

Example 3:

Input: root = []
Output: []
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if root == None:
            return []
        
        q = []
        q.append(root)
        
        res = []
        
        # basic bfs traversal
        while len(q) > 0:
            size = len(q)
            for i in range(size):
                cur = q.pop(0)
                
                if i == size - 1:
                    res.append(cur.val)
                if cur.left != None:
                    q.append(cur.left)
                if cur.right != None:
                    q.append(cur.right)
                    
        return res

```

## Generate a String With Characters That Have Odd Counts

```py
'''
Generate a String With Characters That Have Odd Counts
Easy

Given an integer n, return a string with n characters such that each character in such string occurs an odd number of times.

The returned string must contain only lowercase English letters. If there are multiples valid strings, return any of them.  

 

Example 1:

Input: n = 4
Output: "pppz"
Explanation: "pppz" is a valid string since the character 'p' occurs three times and the character 'z' occurs once. Note that there are many other valid strings such as "ohhh" and "love".

Example 2:

Input: n = 2
Output: "xy"
Explanation: "xy" is a valid string since the characters 'x' and 'y' occur once. Note that there are many other valid strings such as "ag" and "ur".

Example 3:

Input: n = 7
Output: "holasss"
'''


class Solution:
    def generateTheString(self, n: int) -> str:        
        if n % 2 == 0:
            res = "a" * (n - 1)
            res += "b"
            return res
        else:
            res = "a" * n
            return res

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

## Ransom Note

```py
'''
Ransom Note
Easy

Given two stings ransomNote and magazine, return true if ransomNote can be constructed from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.

 

Example 1:

Input: ransomNote = "a", magazine = "b"
Output: false

Example 2:

Input: ransomNote = "aa", magazine = "ab"
Output: false

Example 3:

Input: ransomNote = "aa", magazine = "aab"
Output: true
'''

class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:        
        for char in ransomNote:
            if char not in magazine:
                return False
            
            magazine = magazine.replace(char, "", 1)
            
        return True

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

## Number of Different Integers in a String

```py
'''
Number of Different Integers in a String
Easy

You are given a string word that consists of digits and lowercase English letters.

You will replace every non-digit character with a space. For example, "a123bc34d8ef34" will become " 123  34 8  34". Notice that you are left with some integers that are separated by at least one space: "123", "34", "8", and "34".

Return the number of different integers after performing the replacement operations on word.

Two integers are considered different if their decimal representations without any leading zeros are different.

 

Example 1:

Input: word = "a123bc34d8ef34"
Output: 3
Explanation: The three different integers are "123", "34", and "8". Notice that "34" is only counted once.

Example 2:

Input: word = "leet1234code234"
Output: 2

Example 3:

Input: word = "a1b01c001"
Output: 1
Explanation: The three integers "1", "01", and "001" all represent the same integer because
the leading zeros are ignored when comparing their decimal values.
'''


class Solution:
    def numDifferentIntegers(self, word: str) -> int:
        res = re.sub(r"[a-z]", "-", word)
        
        res = [int(char) for char in res.split("-") if char != "" and char.isnumeric()]
        
        return len(set(res))

```

## Linked List Cycle II

```py
'''
Linked List Cycle II
Medium

Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Notice that you should not modify the linked list.

 

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.

Example 2:

Input: head = [1,2], pos = 0
Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the first node.

Example 3:

Input: head = [1], pos = -1
Output: no cycle
Explanation: There is no cycle in the linked list.
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return None
        
        slow = head
        fast = head
        res = head
        
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            
            if fast is slow:
                res = fast
                break
                
            if fast == None or fast.next == None:
                return None
            
        while res != None and head != None:
            if res == head:
                break
                
            res = res.next
            head = head.next
            
        return res

```

## Maximum Absolute Sum of Any Subarray

```py
'''
Maximum Absolute Sum of Any Subarray
Medium

You are given an integer array nums. The absolute sum of a subarray [numsl, numsl+1, ..., numsr-1, numsr] is abs(numsl + numsl+1 + ... + numsr-1 + numsr).

Return the maximum absolute sum of any (possibly empty) subarray of nums.

Note that abs(x) is defined as follows:

    If x is a negative integer, then abs(x) = -x.
    If x is a non-negative integer, then abs(x) = x.

 

Example 1:

Input: nums = [1,-3,2,3,-4]
Output: 5
Explanation: The subarray [2,3] has absolute sum = abs(2+3) = abs(5) = 5.

Example 2:

Input: nums = [2,-5,1,-4,3,-2]
Output: 8
Explanation: The subarray [-5,1,-4] has absolute sum = abs(-5+1-4) = abs(-8) = 8.
'''


class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return abs(nums[0])
        
        sum1 = 0
        sum2 = 0
        res = 0
        
        for num in nums:
            sum1 += num
            sum2 += num
            sum1 = max(sum1, 0)
            sum2 = min(sum2, 0)
            res = max(res, max(sum1, -1 * sum2))
        
        return res

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

## Unique Morse Code Words

```py
'''
Unique Morse Code Words
Easy

International Morse Code defines a standard encoding where each letter is mapped to a series of dots and dashes, as follows:

    'a' maps to ".-",
    'b' maps to "-...",
    'c' maps to "-.-.", and so on.

For convenience, the full table for the 26 letters of the English alphabet is given below:

[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]

Given an array of strings words where each word can be written as a concatenation of the Morse code of each letter.

    For example, "cab" can be written as "-.-..--...", which is the concatenation of "-.-.", ".-", and "-...". We will call such a concatenation the transformation of a word.

Return the number of different transformations among all words we have.

 

Example 1:

Input: words = ["gin","zen","gig","msg"]
Output: 2
Explanation: The transformation of each word is:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."
There are 2 different transformations: "--...-." and "--...--.".

Example 2:

Input: words = ["a"]
Output: 1
'''


class Solution:
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        dic = {'a':".-",'b':"-...",'c':"-.-.",'d':"-..",'e':".",'f':"..-.",'g':"--.",'h':"....",'i':"..",'j':".---",'k':"-.-",'l':".-..",'m':"--",'n':"-.",'o':"---",'p':".--.",'q':"--.-",'r':".-.",'s':"...",'t':"-",'u':"..-",'v':"...-",'w':".--",'x':"-..-",'y':"-.--",'z':"--.."}
        
        code = []
        
        for word in words:
            t = word
            for c in word:
                t = t.replace(c, dic[c], 1)
            code.append(t)
                    
        return len(set(code))

```

## Fizz Buzz

```py
'''
Fizz Buzz
Easy

Given an integer n, return a string array answer (1-indexed) where:

    answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
    answer[i] == "Fizz" if i is divisible by 3.
    answer[i] == "Buzz" if i is divisible by 5.
    answer[i] == i if non of the above conditions are true.

 

Example 1:

Input: n = 3
Output: ["1","2","Fizz"]

Example 2:

Input: n = 5
Output: ["1","2","Fizz","4","Buzz"]

Example 3:

Input: n = 15
Output: ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
'''

# writing a sick one-liner in python is a different feeling all together

class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        return ["FizzBuzz" if num % 3 == 0 and num % 5 == 0 else "Fizz" if num % 3 == 0 else "Buzz" if num % 5 == 0 else str(num) for num in range(1, n + 1)]

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

## Mean of Array After Removing Some Elements

```py
'''
Mean of Array After Removing Some Elements
Easy

Given an integer array arr, return the mean of the remaining integers after removing the smallest 5% and the largest 5% of the elements.

Answers within 10-5 of the actual answer will be considered accepted.

 

Example 1:

Input: arr = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3]
Output: 2.00000
Explanation: After erasing the minimum and the maximum values of this array, all elements are equal to 2, so the mean is 2.

Example 2:

Input: arr = [6,2,7,5,1,2,0,3,10,2,5,0,5,5,0,8,7,6,8,0]
Output: 4.00000

Example 3:

Input: arr = [6,0,7,0,7,5,7,8,3,4,0,7,8,1,6,8,1,1,2,4,8,1,9,5,4,3,8,5,10,8,6,6,1,0,6,10,8,2,3,4]
Output: 4.77778

Example 4:

Input: arr = [9,7,8,7,7,8,4,4,6,8,8,7,6,8,8,9,2,6,0,0,1,10,8,6,3,3,5,1,10,9,0,7,10,0,10,4,1,10,6,9,3,6,0,0,2,7,0,6,7,2,9,7,7,3,0,1,6,1,10,3]
Output: 5.27778

Example 5:

Input: arr = [4,8,4,10,0,7,1,3,7,8,8,3,4,1,6,2,1,1,8,0,9,8,0,3,9,10,3,10,1,10,7,3,2,1,4,9,10,7,6,4,0,8,5,1,2,1,6,2,5,0,7,10,9,10,3,7,10,5,8,5,7,6,7,6,10,9,5,10,5,5,7,2,10,7,7,8,2,0,1,1]
Output: 5.29167
'''


class Solution:
    def trimMean(self, arr: List[int]) -> float:
        arr.sort()
        
        five_percent = int(0.05 * len(arr))
                
        i = 0
                
        while len(arr) > 0 and i < five_percent:
            arr.pop()
            arr.pop(0)
            i += 1
                                    
        return sum(arr) / len(arr)

```

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

## Valid Anagram

```py
'''
Valid Anagram
Easy

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

 

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true

Example 2:

Input: s = "rat", t = "car"
Output: false
'''


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        dic = {}
        
        for char in s:
            if char in dic:
                dic[char] += 1
            else:
                dic[char] = 1
                
        for char in t:
            if char not in dic or dic[char] < 1:
                return False
            
            dic[char] -= 1
            
        return True

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

