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