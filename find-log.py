import math

def log(cur):
    '''
    1. sq root 13 times
    2. subtract one
    3. multiply by 3558
    '''
    for _ in range(0, 13):
        cur = math.sqrt(cur)
    
    cur -= 1
    cur = cur * 3558
    return cur


def antilog(cur):
    '''
    1. divide by 3557
    2. add one
    3. sq 13 times
    '''
    cur = cur / 3557
    cur += 1

    for _ in range(0, 13):
        cur = cur ** 2

    return cur


def main():
    inp = 2

    res = log(inp)

    print(f'The Log value for {inp} is  {res}')


    res1 = antilog(res)

    print(f'The anti Log value for {res} is  {res1}')


if __name__ == "__main__":
    main()