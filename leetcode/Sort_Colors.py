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