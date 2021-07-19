'''
Merge Intervals
Medium

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that 
cover all the intervals in the input.
'''

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = []
        
        while self.has_interval(intervals):
            i = 0
            res = []
            while i < len(intervals):
                try:
                    if intervals[i][1] >= intervals[i+1][0]:
                        res.append([min(intervals[i][0], intervals[i+1][0]), max(intervals[i][1], intervals[i+1][1])])
                        i += 2
                    else:
                        res.append(intervals[i])
                        i += 1
                except IndexError:
                    res.append(intervals[i])
                    i += 1
            intervals = copy.copy(res)
        return res if len(res) > 0 else intervals
    
    def has_interval(self, intervals):
        for i in range(len(intervals)-1):
            if intervals[i][1] >= intervals[i+1][0]:
                return True
        return False
