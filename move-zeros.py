def move_zeros(nums):
    count = 0

    for i in nums:
        if i == 0:
            nums.remove(i)
            count += 1
    
    while count > 0:
        nums.append(0)
        count -= 1
    return nums


def main():
    inp = [1,3,6,0,6,4,8,4,0,5,3,0]

    res = move_zeros(inp)

    print(f'new arr = {res}')


if __name__ == "__main__":
    main()