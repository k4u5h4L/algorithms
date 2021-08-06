def floyd(n, r):	
    for k in range(n):
        for i in range(n):
            for j in range(n):
                r[i][j] = min(r[i][j], r[i][k] + r[k][j])

    return r

def main():

    n = 3

    mat = [
        [1,2,3],
        [6,5,2],
        [5,1,4]
    ]

    res = floyd(n, mat)

    print("The answer is:")
    print(res)


if __name__ == "__main__":
    main()

