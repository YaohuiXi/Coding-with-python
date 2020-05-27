## 1、矩阵
### 1.1 LeetCode64 矩阵的最小路径和
法二：压缩空间，一维数组做
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # 一维数组
        m = len(grid)
        n = len(grid[0])      
        dp = [0]*n
        # 初始化第一行
        dp[0] = grid[0][0]
        for k in range(1,n):
            dp[k] = dp[k-1] + grid[0][k]
        for i in range(1,m):
            for j in range(0,n):
                if j == 0:
                    dp[j] += grid[i][0]
                else:
                    dp[j] = min(dp[j],dp[j-1])+grid[i][j]
        return dp[-1]
```
### 1.2 矩阵的总路径数
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        cur = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                cur[j] += cur[j-1]
        return cur[-1]
```
## 2、子序列
### 2.1 最长递增子序列
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        dp = [1] * n
        for i in range(1,n):
            for j in range(0,i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)
```
### 2.2 最长公共子序列
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1 = len(text1)
        n2 = len(text2)
        dp = [[0]*(n1+1) for _ in range(n2+1)]
        # print(dp)
        for i in range(1,n2+1):
            for j in range(1,n1+1):
                if text1[j-1] ==text2[i-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]
```
## 3、0-1背包
### 3.1 划分数组为和相等的两部分
```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        nums_sum = sum(nums)
        if nums_sum % 2 != 0:
            return False
        c =  nums_sum //2
        dp = [[0] * (c+1) for _ in range(len(nums))]
        for i in range(len(nums)):
            dp[i][0] = 1
        if nums[0] < c:
            dp[0][nums[0]] = 1
        for i in range(1,len(nums)):
            num = nums[i]
            for j in range(1,c+1):
                if num > j:
                    dp[i][j] = dp[i-1][j]
                    continue
                dp[i][j] = dp[i-1][j] or dp[i-1][j-num]
        return dp[-1][-1]
```
### 3.2 找零钱的最少硬币数目
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        memo = [0]+ [amount+1] * (amount)
        for i in range(1,amount+1):
            for coin in coins:
                if coin <= i:
                    memo[i] = min(memo[i],memo[i-coin]+1)
        if memo[-1] == amount+1:
            return -1
        else:
            return memo[-1]
```
### 3.3 找零钱的硬币数组合
```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount+1)
        dp[0] = 1
        for coin in coins:
            for i in range(1,amount+1):
                if i >= coin:
                    dp[i] += dp[i-coin]
        return dp[-1]
```
## 4、房屋打劫（3题）

## 5、股票交易（6题）


