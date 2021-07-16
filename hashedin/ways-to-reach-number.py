def total_ways(start, end, memo={}):
    if start in memo:
        return memo[start]
    if start == end:
        return 1
    elif start > end:
        return 0
    else:
        memo[start] = total_ways(start+1, end, memo) + total_ways(start+2, end, memo)
        return memo[start]

def main():
    start = 3
    end = 5

    # using a dynamic programming approach to reduce recursion calls
    res = total_ways(start, end)

    print(f'total ways from {start} to {end} is  {res}')


if __name__ == "__main__":
    main()