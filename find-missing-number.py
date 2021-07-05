def find_missing_no_indices(nums):
    '''
    constraint is that the numbers are first n+1. which means that they start from 1. 
    so we can use their indices as hash keys to see if they exist
    '''

def find_missing_no_sorted(nums):
    '''
    constraint is that the numbers are sorted. 
    so a simple binary search will do
    '''

    left = 0
    right = len(nums) - 1

    while left < right:
        mid = int(left + (right - left) / 2)

        if nums[mid] != nums[mid + 1]:
            return nums[mid] + 1
        # elif nums[mid]

def find_missing_no_linear(nums):
    '''
    simple linear search
    '''
    for i in range(len(nums)-1):
        if nums[i]+1 != nums[i+1]:
            return nums[i] + 1
    return -1


def main():
    inp = [1, 2, 3, 4, 6]

    res = find_missing_no_linear(inp)

    print(f'missing number in {inp} is   {res}')


if __name__ == "__main__":
    main()