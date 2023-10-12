
# leetcode programs:
## Page: 1
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

## Find Words That Can Be Formed by Characters

```py
'''
Find Words That Can Be Formed by Characters
Easy

You are given an array of strings words and a string chars.

A string is good if it can be formed by characters from chars (each character can only be used once).

Return the sum of lengths of all good strings in words.

 

Example 1:

Input: words = ["cat","bt","hat","tree"], chars = "atach"
Output: 6
Explanation: The strings that can be formed are "cat" and "hat" so the answer is 3 + 3 = 6.

Example 2:

Input: words = ["hello","world","leetcode"], chars = "welldonehoneyr"
Output: 10
Explanation: The strings that can be formed are "hello" and "world" so the answer is 5 + 5 = 10.
'''


class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:  
        res = []
        
        for word in words:
            temp = word
            
            for c in chars:
                temp = temp.replace(c, "", 1)
                
                if temp == "":
                    res.append(word)
                    break
                                        
        length = 0
        
        for word in res:
            length += len(word)
            
        return length

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

## Last Stone Weight

```py
'''
Last Stone Weight
Easy

We have a collection of stones, each stone has a positive integer weight.

Each turn, we choose the two heaviest stones and smash them together.  Suppose the stones have weights x and y with x <= y.  The result of this smash is:

    If x == y, both stones are totally destroyed;
    If x != y, the stone of weight x is totally destroyed, and the stone of weight y has new weight y-x.

At the end, there is at most 1 stone left.  Return the weight of this stone (or 0 if there are no stones left.)

 

Example 1:

Input: [2,7,4,1,8,1]
Output: 1
Explanation: 
We combine 7 and 8 to get 1 so the array converts to [2,4,1,1,1] then,
we combine 2 and 4 to get 2 so the array converts to [2,1,1,1] then,
we combine 2 and 1 to get 1 so the array converts to [1,1,1] then,
we combine 1 and 1 to get 0 so the array converts to [1] then that's the value of last stone
'''


class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        if len(stones) == 1:
            return stones[0]
        
        stones.sort(reverse=True)
        
        while len(stones) >= 2:
            stone1 = stones.pop(0)
            stone2 = stones.pop(0)
            
            if stone1 == stone2:
                continue
            else:
                stones.append(abs(stone1 - stone2))
                stones.sort(reverse=True)
                
        if len(stones) == 1:
            return stones[0]
        else:
            return 0

```

## Pascal's Triangle

```py
'''
Pascal's Triangle
Easy

Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

 

Example 1:

Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

Example 2:

Input: numRows = 1
Output: [[1]]
'''


class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        triangle = []

        for row_num in range(numRows):
            row = [None for _ in range(row_num + 1)]
            row[0], row[-1] = 1, 1

            for j in range(1, len(row) - 1):
                row[j] = triangle[row_num - 1][j - 1] + triangle[row_num - 1][j]

            triangle.append(row)

        return triangle

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

## Maximum 69 Number

```py
'''
Maximum 69 Number
Easy

Given a positive integer num consisting only of digits 6 and 9.

Return the maximum number you can get by changing at most one digit (6 becomes 9, and 9 becomes 6).

 

Example 1:

Input: num = 9669
Output: 9969
Explanation: 
Changing the first digit results in 6669.
Changing the second digit results in 9969.
Changing the third digit results in 9699.
Changing the fourth digit results in 9666. 
The maximum number is 9969.

Example 2:

Input: num = 9996
Output: 9999
Explanation: Changing the last digit 6 to 9 results in the maximum number.

Example 3:

Input: num = 9999
Output: 9999
Explanation: It is better not to apply any change.
'''

class Solution:
    def maximum69Number (self, num: int) -> int:
        num = str(num)
        
        res = ""
        
        found = False
        
        for char in num:
            if char == '6' and found == False:
                res += '9'
                found = True
            else:
                res += char
                
        return int(res)

```

## Longest Word in Dictionary

```py
'''
Longest Word in Dictionary
Medium

Given an array of strings words representing an English Dictionary, return the longest word in words that can be built one character at a time by other words in words.

If there is more than one possible answer, return the longest word with the smallest lexicographical order. If there is no answer, return the empty string.

 

Example 1:

Input: words = ["w","wo","wor","worl","world"]
Output: "world"
Explanation: The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".

Example 2:

Input: words = ["a","banana","app","appl","ap","apply","apple"]
Output: "apple"
Explanation: Both "apply" and "apple" can be built from other words in the dictionary. However, "apple" is lexicographically smaller than "apply".
'''


class Solution:
    def longestWord(self, words: List[str]) -> str:
        hs = set(words)
        
        words.sort(key=lambda x: len(x), reverse=True)
        
        res = []
        for word in words:
            i = 1
            hs.remove(word)
            while i < len(word):
                if word[:i] not in hs:
                    break
                i += 1
            
            if i == len(word):
                res.append(word)
            
            hs.add(word)
        
        max_len = 0
        
        for word in res:
            max_len = max(max_len, len(word))
        
        final_res = []
        
        for word in res:
            if len(word) == max_len:
                final_res.append(word)
        
        final_res.sort()
        
        return final_res[0] if len(final_res) != 0 else ""

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

## ZigZag Conversion

```py
'''
ZigZag Conversion
Medium

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R

And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);

 

Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I

Example 3:

Input: s = "A", numRows = 1
Output: "A"
'''


class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if len(s) == 1 or numRows == 1:
            return s
            
        zig_zag = []
        
        for _ in range(numRows):
            zig_zag.append([" "] * (len(s)))
            
        i = 0
        j = 0
        point = 0
        
        down_or_slant = True
        
        while point < len(s):
            if down_or_slant:
                while i < numRows and point < len(s):
                    zig_zag[i][j] = s[point]
                    i += 1
                    point += 1
                    
                down_or_slant = not down_or_slant
                j += 1
                
            else:
                i -= 2
                
                while i >= 0 and point < len(s):
                    zig_zag[i][j] = s[point]
                    i -= 1
                    j += 1

                    point += 1
                    
                i += 2
                
                down_or_slant = not down_or_slant
                
        res = ""
        
        for i in range(len(zig_zag)):
            for j in range(len(zig_zag[i])):
                if zig_zag[i][j] != " ":
                    res += zig_zag[i][j]

        return res

```

## Kth Largest Element in an Array

```py
'''
Kth Largest Element in an Array
Medium

Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

 

Example 1:

Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Example 2:

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
'''


import queue

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        q = queue.PriorityQueue()
        
        for num in nums:
            q.put(num)
            
        for _ in range(len(nums) - k):
            temp = q.get()
        
        return q.get()

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

## Best Time to Buy and Sell Stock II

```py
'''
Best Time to Buy and Sell Stock II
Easy

You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.

Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.

Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e., max profit = 0.
'''


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        
        for i in range(len(prices) - 1):
            if prices[i+1] > prices[i]:
                profit += prices[i+1] - prices[i]
            
        return profit

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

## Matrix Diagonal Sum

```py
'''
Matrix Diagonal Sum
Easy

Given a square matrix mat, return the sum of the matrix diagonals.

Only include the sum of all the elements on the primary diagonal and all the elements on the secondary diagonal that are not part of the primary diagonal.

 

Example 1:

Input: mat = [[1,2,3],
              [4,5,6],
              [7,8,9]]
Output: 25
Explanation: Diagonals sum: 1 + 5 + 9 + 3 + 7 = 25
Notice that element mat[1][1] = 5 is counted only once.

Example 2:

Input: mat = [[1,1,1,1],
              [1,1,1,1],
              [1,1,1,1],
              [1,1,1,1]]
Output: 8

Example 3:

Input: mat = [[5]]
Output: 5
'''

class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        m = len(mat)
        
        if m == 1:
            return mat[0][0]

        res = 0
        
        for i in range(m):
           res += mat[i][i]
           res += mat[i][-1 - i]

        if m % 2 == 1:
            res -= mat[m // 2][m // 2]

        return res

```

## Reverse Integer

```py
'''
Reverse Integer
Easy

Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

 

Example 1:

Input: x = 123
Output: 321

Example 2:

Input: x = -123
Output: -321

Example 3:

Input: x = 120
Output: 21

Example 4:

Input: x = 0
Output: 0
'''


class Solution:
    def reverse(self, x: int) -> int:
        neg = False
        
        if x == 0:
            return x
        
        elif x < 0:
            neg = True
            x = abs(x)
            
        n = str(x)
        
        n = n[::-1]
                
        if neg:
            n = f"-{n}"
            
        n = int(n)
        
        if n < (-2 ** 31) or n > (2 ** 31) - 1:
            return 0
        
        else:
            return n

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

## Occurrences After Bigram

```py
'''
Occurrences After Bigram
Easy

Given two strings first and second, consider occurrences in some text of the form "first second third", where second comes immediately after first, and third comes immediately after second.

Return an array of all the words third for each occurrence of "first second third".

 

Example 1:

Input: text = "alice is a good girl she is a good student", first = "a", second = "good"
Output: ["girl","student"]

Example 2:

Input: text = "we will we will rock you", first = "we", second = "will"
Output: ["we","rock"]
'''



class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        words = text.split(" ")
        
        if len(words) < 3:
            return []
        
        res = []
        
        for i in range(len(words) - 2):
            if words[i] == first and words[i+1] == second:
                res.append(words[i+2])
                
        return res

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

## Implement Queue using Stacks

```py
'''
Implement Queue using Stacks
Easy

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:

    void push(int x) Pushes element x to the back of the queue.
    int pop() Removes the element from the front of the queue and returns it.
    int peek() Returns the element at the front of the queue.
    boolean empty() Returns true if the queue is empty, false otherwise.

Notes:

    You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
    Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.

Follow-up: Can you implement the queue such that each operation is amortized O(1) time complexity? In other words, performing n operations will take overall O(n) time even if one of those operations may take longer.

 

Example 1:

Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]

Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false

 

Constraints:

    1 <= x <= 9
    At most 100 calls will be made to push, pop, peek, and empty.
    All the calls to pop and peek are valid.
'''


class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = []
        self.s2 = []
        

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        def s2_empty():
            return len(self.s2) == 0
            
        self.s2 = []
        
        while not self.empty():
            self.s2.append(self.s1.pop())
            
        self.s1.append(x)
        
        while not s2_empty():
            self.s1.append(self.s2.pop())
        

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        return self.s1.pop()
        

    def peek(self) -> int:
        """
        Get the front element.
        """
        return self.s1[-1]
        

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.s1) == 0
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()

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

## Count the Number of Consistent Strings

```py
'''
Count the Number of Consistent Strings
Easy

You are given a string allowed consisting of distinct characters and an array of strings words. A string is consistent if all characters in the string appear in the string allowed.

Return the number of consistent strings in the array words.

 

Example 1:

Input: allowed = "ab", words = ["ad","bd","aaab","baa","badab"]
Output: 2
Explanation: Strings "aaab" and "baa" are consistent since they only contain characters 'a' and 'b'.

Example 2:

Input: allowed = "abc", words = ["a","b","c","ab","ac","bc","abc"]
Output: 7
Explanation: All strings are consistent.

Example 3:

Input: allowed = "cad", words = ["cc","acd","b","ba","bac","bad","ac","d"]
Output: 4
Explanation: Strings "cc", "acd", "ac", and "d" are consistent.
'''

class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        count = 0
        for word in words:
            if self.is_present(word, allowed):
                count += 1
                
        return count
    
    def is_present(self, word, allowed):
        for char in word:
            if char not in allowed:
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

## Minimum Absolute Difference in BST

```py
'''
Minimum Absolute Difference in BST
Easy

Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.

 

Example 1:

Input: root = [4,2,6,1,3]
Output: 1

Example 2:

Input: root = [1,0,48,null,null,12,49]
Output: 1
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        vals = []
        
        self.inorder(root, vals)
        
        res = max(vals)
                
        for i in range(len(vals)-1):
            res = min(res, abs(vals[i] - vals[i+1]))
                
        return res
    
    def inorder(self, root, vals):
        if root == None:
            return
        
        self.inorder(root.left, vals)
        vals.append(root.val)
        self.inorder(root.right, vals)

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

