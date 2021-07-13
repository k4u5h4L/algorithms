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