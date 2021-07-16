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
