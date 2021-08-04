'''
Coin Change
Medium

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:

Input: coins = [2], amount = 3
Output: -1

Example 3:

Input: coins = [1], amount = 0
Output: 0

Example 4:

Input: coins = [1], amount = 1
Output: 1

Example 5:

Input: coins = [1], amount = 2
Output: 2
'''

class Solution:
    def coinChange(self, coins, amount):
        if len(coins) == 1:
            if amount % coins[0] == 0:
                return amount // coins[0]
            else:
                return -1
        elif amount == 0:
            return 0
                
        coins.sort(reverse=True)
        
        res = []
                    
        self.num_coins(coins, amount, 0, res)
                
        return min(res)
    
    def num_coins(self, coins, amount, steps, res, memo={}):
        if amount in memo:
            if len(res) == 0 or steps < min(res):
                res.append(steps + memo[amount])
            return
        if amount == 0:
            if len(res) == 0 or steps < min(res):
                res.append(steps)
            return
        
        elif amount < 0:
            return
        
        else:
            for coin in coins:
                if amount - coin >= 0:
                    if amount - coin == 0:
                        memo[amount] = steps+1
                    self.num_coins(coins, amount - coin, steps+1, res, memo)
                    
            return


res = Solution()

coins = [1,2,5]
amt = 50

print(res.coinChange(coins, amt))