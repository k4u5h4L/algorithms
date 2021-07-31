'''
For the given array of length and price find the maximum profit that can be gained.
'''

def maxProfit(prices):
        max_profit = 0
        min_price = prices[0]
        for price in prices:
            min_price = min(price, min_price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

def main():
    prices = [2,4,2,5,8,2,3]

    res = maxProfit(prices)

    print(f"Max profit of {prices} is {res}")

if __name__ == "__main__":
    main()