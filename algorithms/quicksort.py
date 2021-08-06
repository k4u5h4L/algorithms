def quicksort(a, first, last):
    if first < last:
        pivot = first
        i = first
        j = last

        while i < j:
            while a[i] <= a[pivot] and i < last:
                i += 1
            while a[j] > a[pivot]:
                j -= 1

            if i < j:
                a[i], a[j] = a[j], a[i]

        a[pivot], a[j] = a[j], a[pivot]

        quicksort(a, first, j - 1);
        quicksort(a, j + 1, last);

def main():
    arr = [5,3,4,1,2]

    quicksort(arr, 0, len(arr) - 1)

    print("The answer is:")
    print(arr)


if __name__ == "__main__":
    main()