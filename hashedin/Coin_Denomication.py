'''
Other program here called Coin_Change.py is better suited for this. Below is a greedy method, is not according to the question given
'''

def num_coins(num, coins):
    dic = {}

    for coin in coins:
        dic[coin] = 0

    top = len(coins) - 1
    while num > 0:
        if num - coins[top] >= 0:
            dic[coins[top]] += 1
            num -= coins[top]
        else:
            top -= 1

    return dic


def main():
    number = 978

    coins = [1,2,5,10,50,100]

    res = num_coins(number, coins)

    print(f"The number {number} can be split into {res}")

if __name__ == "__main__":
    main()
