'''
For the given array with n numbers find the maximum value of x Xor for a given x.
'''

def max_xor(arr, key):
    xor = 0

    for a in arr:
        xor = max(xor, key ^ a)

    return xor

def main():
    arr = [1,4,2,5,7,1,3,5,2,3,1]
    key = 3

    res = max_xor(arr, 3)

    print(f"MAx XOR of {key} is {res}")


if __name__ == "__main__":
    main()