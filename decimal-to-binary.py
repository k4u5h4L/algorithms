def getBin(decimal):
    result = []
    while decimal > 0:
        if decimal % 2 == 0:
            result.append("0")
        else:
            result.append("1")
        decimal = int(decimal / 2)

    result.reverse()
    result = ''.join(result)
    return result


def main():
    inp = 6

    res = getBin(inp)

    print(f'Binary for {inp}  is  {res}')


if __name__ == "__main__":
    main()