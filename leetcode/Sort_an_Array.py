'''
Sort an Array
Medium

Given an array of integers nums, sort the array in ascending order.
'''

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.quicksort(nums, 0, len(nums) - 1);
        return nums
        
    def quicksort(self, a, first, last):
        i = 0
        j = 0
        pivot = 0

        if first < last:
            pivot = first
            i = first
            j = last

            while (i < j):
                while a[i] <= a[pivot] and i < last:
                    i += 1
                while a[j] > a[pivot]:
                    j -= 1

                if i < j:
                    temp = a[i]
                    a[i] = a[j]
                    a[j] = temp

            temp = a[pivot]
            a[pivot] = a[j]
            a[j] = temp

            self.quicksort(a, first, j - 1);
            self.quicksort(a, j + 1, last);