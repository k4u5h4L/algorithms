from itertools import permutations

def count_anagrams(s, t):
    s_perm = list(permutations([char for char in s]))

    ways = []
    
    for perm in s_perm:
        ways.append(''.join(perm))
    
    count = 0

    for word in ways:
        if word in t:
            count += 1

    return count


def main():
    s = "for"
    t = "forxxorfxdofr"

    res = count_anagrams(s, t)

    print(f"Number of anagrams of {s} in {t} is  {res}")


if __name__ == "__main__":
    main()