from itertools import permutations

# you can use permutations, or check each section
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
# ----------------------------------------------------------------------------------------------------------
# check if string is anagram of each other
def is_anagram(s, s1):
    s = list(s)
    s1 = list(s1)

    s.sort()
    s1.sort()

    return s == s1

# send each subsection to be checked
def count_anagrams_2(s, t):
    res = 0

    for i in range(len(t) - len(s)+1):
        print(t[i:i+len(s)])
        if is_anagram(s, t[i:i+len(s)]):
            res += 1


    return res

# ----------------------------------------------------------------------------------------------------------



def main():
    s = "for"
    t = "forxxorfxdofr"

    res = count_anagrams(s, t)

    print(f"Number of anagrams of {s} in {t} is  {res}")


if __name__ == "__main__":
    main()
