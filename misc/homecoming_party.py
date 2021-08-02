'''
The original family in the town has organised a homecoming party with N people invited. 
Each person has a special trust value denoted by array A. 
A person i will be friends with a person j only if either A[i] % A[j]==0 or A[j] % A[i]==0. 
Find the maximum number of friends each person can make in this party.


Problem Constraints:
1 <= N <= 2 * 10 ^ 5
1 <= A[i] <= 10 ^ 5


Input Format:
1st and only arguement has an integer array A

Output Format:
Return an integer array containing number of freind of each person. 

Example Input 
Input1:
1 = [2, 3, 4, 5, 6]
'''

import time

def number_of_friends(arr):
    res = [0] * len(arr)

    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] % arr[j] == 0 or arr[j] % arr[i] == 0:
                res[i] += 1

    return res


def main():
    arr = [2, 3, 4, 5, 6]

    start_time = time.time()
    friends = number_of_friends(arr)
    end_time = time.time()

    time_taken = str(end_time - start_time)
    time_taken = time_taken[:time_taken.index('.')+3]

    print(f"Number of friends of {arr} is {friends}")
    print(f"Time taken: {time_taken} ms")

main()