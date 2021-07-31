'''
For the given element in an array find the frequency of that element.
'''

def freq(arr, key):
    max_freq = 0

    for a in arr:
        if a == key:
            max_freq += 1

    return max_freq

def main():
    arr = [1,4,2,5,7,1,3,5,2,3,1]
    key = 3

    res = freq(arr, 3)

    print(f"Frequency of {key} is {res}")


if __name__ == "__main__":
    main()