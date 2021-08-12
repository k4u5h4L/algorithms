
# hashedin programs:

## find-log

```py
import math

def log(cur):
    '''
    1. sq root 13 times
    2. subtract one
    3. multiply by 3558
    '''
    for _ in range(0, 13):
        cur = math.sqrt(cur)
    
    cur -= 1
    cur = cur * 3558
    return cur


def antilog(cur):
    '''
    1. divide by 3557
    2. add one
    3. sq 13 times
    '''
    cur = cur / 3557
    cur += 1

    for _ in range(0, 13):
        cur = cur ** 2

    return cur


def main():
    inp = 2

    res = log(inp)

    print(f'The Log value for {inp} is  {res}')


    res1 = antilog(res)

    print(f'The anti Log value for {res} is  {res1}')


if __name__ == "__main__":
    main()
```

## minimum platforms required

```py
# Program to find minimum
# number of platforms
# required on a railway
# station

# Code got from geeksforgeeks

# Returns minimum number
# of platforms required


def findPlatform(arr, dep, n):

	# Sort arrival and
	# departure arrays
	arr.sort()
	dep.sort()

	# plat_needed indicates
	# number of platforms
	# needed at a time
	plat_needed = 1
	result = 1
	i = 1
	j = 0

	# Similar to merge in
	# merge sort to process
	# all events in sorted order
	while (i < n and j < n):

		# If next event in sorted
		# order is arrival,
		# increment count of
		# platforms needed
		if (arr[i] <= dep[j]):

			plat_needed += 1
			i += 1

		# Else decrement count
		# of platforms needed
		elif (arr[i] > dep[j]):

			plat_needed -= 1
			j += 1

		# Update result if needed
		if (plat_needed > result):
			result = plat_needed

	return result

# Driver code


arr = [900, 940, 950, 1100, 1500, 1800]
dep = [910, 1200, 1120, 1130, 1900, 2000]
n = len(arr)

print("Minimum Number of Platforms Required = ",
	findPlatform(arr, dep, n))

```

## find-missing-number

```py
def find_missing_no_indices(nums):
    '''
    constraint is that the numbers are first n+1. which means that they start from 1. 
    so we can use their indices as hash keys to see if they exist
    '''

def find_missing_no_sorted(nums):
    '''
    constraint is that the numbers are sorted. 
    so a simple binary search will do
    '''

    left = 0
    right = len(nums) - 1

    while left < right:
        mid = int(left + (right - left) / 2)

        if nums[mid] != nums[mid + 1]:
            return nums[mid] + 1
        # elif nums[mid]

def find_missing_no_linear(nums):
    '''
    simple linear search
    '''
    for i in range(len(nums)-1):
        if nums[i]+1 != nums[i+1]:
            return nums[i] + 1
    return -1


def main():
    inp = [1, 2, 3, 4, 6]

    res = find_missing_no_linear(inp)

    print(f'missing number in {inp} is   {res}')


if __name__ == "__main__":
    main()
```

## ways-to-reach-number

```py
def total_ways(start, end, memo={}):
    if start in memo:
        return memo[start]
    if start == end:
        return 1
    elif start > end:
        return 0
    else:
        memo[start] = total_ways(start+1, end, memo) + total_ways(start+2, end, memo)
        return memo[start]

def main():
    start = 3
    end = 5

    # using a dynamic programming approach to reduce recursion calls
    res = total_ways(start, end)

    print(f'total ways from {start} to {end} is  {res}')


if __name__ == "__main__":
    main()
```

## sqrt

```py

def Square(n, i, j):
 
    mid = (i + j) / 2
    mul = mid * mid
 
    # If mid itself is the square root,
    # return mid
    if ((mul == n) or (abs(mul - n) < 0.00001)):
        return mid
 
    # If mul is less than n, recur second half
    elif (mul < n):
        return Square(n, mid, j)
 
    # Else recur first half
    else:
        return Square(n, i, mid)
 
# Function to find the square root of n
def findSqrt(n):
    i = 1
 
    # While the square root is not found
    found = False
    while (found == False):
 
        # If n is a perfect square
        if (i * i == n):
            print(i)
            found = True
         
        elif (i * i > n):
 
            # Square root will lie in the
            # interval i-1 and i
            res = Square(n, i - 1, i);
            print ("{0:.5f}".format(res))
            found = True
        i += 1


def main():
    num = int(input("Enter a number: "))

    findSqrt(num)

    # print(f'The square root of {num} is  {res}')




if __name__ == "__main__":
    main()
```

## decimal-to-binary

```py
def getBin(decimal):
    result = []
    while decimal > 0:
        if decimal % 2 == 0:
            result.append("0")
        else:
            result.append("1")
        decimal = int(decimal / 2)

    result.reverse()
    result = ''.join(result)
    return result


def main():
    inp = 6

    res = getBin(inp)

    print(f'Binary for {inp}  is  {res}')


if __name__ == "__main__":
    main()
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

## anagram count

```py
from itertools import permutations

def count_anagrams(s, t):
    s_perm = list(permutations([char for char in s]))

    ways = []
    
    for perm in s_perm:
        ways.append(''.join(perm))
    
    count = 0

    for word in ways:
        if word in t:
            count += 1

    return count


def main():
    s = "for"
    t = "forxxorfxdofr"

    res = count_anagrams(s, t)

    print(f"Number of anagrams of {s} in {t} is  {res}")


if __name__ == "__main__":
    main()
```

## move-zeros

```py
def move_zeros(nums):
    count = 0

    for i in nums:
        if i == 0:
            nums.remove(i)
            count += 1
    
    while count > 0:
        nums.append(0)
        count -= 1
    return nums


def main():
    inp = [1,3,6,0,6,4,8,4,0,5,3,0]

    res = move_zeros(inp)

    print(f'new arr = {res}')


if __name__ == "__main__":
    main()
```

## Coin Denomication

```py
'''
Other program here called Coin_Change.py is better suited for this. Below is a greedy method, is not according to the question given
'''

def num_coins(num, coins):
    dic = {}

    for coin in coins:
        dic[coin] = 0

    top = len(coins) - 1
    while num > 0:
        if num - coins[top] >= 0:
            dic[coins[top]] += 1
            num -= coins[top]
        else:
            top -= 1

    return dic


def main():
    number = 978

    coins = [1,2,5,10,50,100]

    res = num_coins(number, coins)

    print(f"The number {number} can be split into {res}")

if __name__ == "__main__":
    main()

```

