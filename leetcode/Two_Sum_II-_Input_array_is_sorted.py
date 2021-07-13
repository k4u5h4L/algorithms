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