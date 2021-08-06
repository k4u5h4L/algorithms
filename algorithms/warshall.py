def warshall(n, r):	
    for k in range(n):
        for i in range(n):
            for j in range(n):
                r[i][j] = r[i][j] or r[i][k] and r[k][j]

    return r

def main():

    n = 3

    mat = [
        [1,0,0],
        [1,0,1],
        [0,1,0]
    ]

    res = warshall(n, mat)

    print("The answer is:")
    print(res)


if __name__ == "__main__":
    main()

