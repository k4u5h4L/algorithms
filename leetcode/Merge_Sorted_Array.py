'''
 Merge Sorted Array
Easy

You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of 
elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. 
To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 
and should be ignored. nums2 has a length of n.
'''

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums = nums1.copy()
        l1 = 0
        l2 = 0
        top = 0
        while l1 < m and l2 < n:
            if nums[l1] < nums2[l2]:
                nums1[top] = nums[l1]
                l1 += 1
            else:
                nums1[top] = nums2[l2]
                l2 += 1
            top += 1
        
        while l1 < m:
            nums1[top] = nums[l1]
            l1 += 1
            top += 1
        
        while l2 < n:
            nums1[top] = nums2[l2]
            l2 += 1
            top += 1
