'''
Average Salary Excluding the Minimum and Maximum Salary
Easy

Given an array of unique integers salary where salary[i] is the salary of the employee i.

Return the average salary of employees excluding the minimum and maximum salary.
'''

class Solution:
    def average(self, salary: List[int]) -> float:
        salary.pop(salary.index(min(salary)))
        salary.pop(salary.index(max(salary)))
        
        return sum(salary) / len(salary)