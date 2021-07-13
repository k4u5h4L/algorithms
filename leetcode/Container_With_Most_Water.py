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