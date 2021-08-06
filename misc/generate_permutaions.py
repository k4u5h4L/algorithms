# Function for generating different permutations of the string
def generatePermutation(string, start, end):
    current = 0
    # Prints the permutations
    if start == end - 1:
        print(string)
    
    else: 
        for current in range(start, end):

            # Swapping the string by fixing a character

            x = list(string)

            x[start], x[current] = x[current], x[start]

            # Recursively calling function generatePermutation() for rest of the characters

            generatePermutation("".join(x), start + 1, end)

            # Swapping the string by fixing a character

            x[start], x[current] = x[current], x[start]

str1 = "ABC"
n = len(str1)
print("All the permutations of the string are: ")
generatePermutation(str1, 0, n)