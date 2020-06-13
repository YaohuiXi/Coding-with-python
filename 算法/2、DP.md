## 1、矩阵[经典]
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
## 2、子序列问题【经典】
### leetcode 300 最长递增子序列
法一：两层循环，复杂度O(n^2)
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
法二：优化，复杂度O(nlogn)

参考资料：https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/dong-tai-gui-hua-er-fen-cha-zhao-tan-xin-suan-fa-p/

第 1 步：定义新状态（特别重要）

tail[i] 表示长度为 i + 1 的所有上升子序列的结尾的最小值。

第 2 步：思考状态转移方程

数组 tail 也是一个严格上升数组

* 流程：
1、设置一个数组 tail，初始时为空；

2、在遍历数组 nums 的过程中，每来一个新数 num，如果这个数严格大于有序数组 tail 的最后一个元素，就把 num 放在有序数组 tail 的后面，否则进入第 3 点；

3、在有序数组 tail 中查找第 1 个等于大于 num 的那个数，试图让它变小；

4、遍历完整个数组 nums，最终有序数组 tail 的长度，就是所求的“最长上升子序列”的长度。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        size = len(nums)
        # 特判
        if size < 2:
            return size
        # 为了防止后序逻辑发生数组索引越界，先把第 1 个数放进去
        tail = [nums[0]]
        for i in range(1, size):
            # 【逻辑 1】比 tail 数组实际有效的末尾的那个元素还大
            # 先尝试是否可以接在末尾
            if nums[i] > tail[-1]:
                tail.append(nums[i])
                continue
            # 使用二分查找法，在有序数组 tail 中
            # 找到第 1 个大于等于 nums[i] 的元素，尝试让那个元素更小
            left = 0
            right = len(tail) - 1
            while left < right:
                # 选左中位数不是偶然，而是有原因的，原因请见 LeetCode 第 35 题题解
                mid = left + (right - left) // 2
                # mid = (left + right) >> 1
                if tail[mid] < nums[i]:
                    # 中位数肯定不是要找的数，把它写在分支的前面
                    left = mid + 1
                else:
                    right = mid
            # 走到这里是因为【逻辑 1】的反面，因此一定能找到第 1 个大于等于 nums[i] 的元素，因此无需再单独判断
            tail[left] = nums[i]
        return len(tail)
```

### leetcode 376. 摆动序列（复杂度O(n)）
法一：类似最长递增，两重循环做，复杂度O(n^2)
```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return len(nums)
        memo = [[1,1] for _ in range (len(nums))]
        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    # print(memo[i])
                    memo[i][0] = max(memo[i][0],memo[j][1]+1)
                if nums[i] < nums[j]:
                    memo[i][1] = max(memo[i][1],memo[j][0]+1)
        return max(memo[-1])
```
法二：DP


法三：贪心，复杂度O(n)
```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2: return n
        up, down = 1,1
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                # 这次上升=到上次下降+1
                up = down + 1
            elif nums[i] < nums[i - 1]:
                # 这次下降=到上次上升+1
                down = up + 1
        return max(up,down)
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

提高版
* LeetCode 84 柱状图中最大的矩形
https://leetcode-cn.com/problems/largest-rectangle-in-histogram/

之前忘了哪家笔试考了，完全做不出来，LeetCode真的非常有用啊
```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        left, right = [0] * n, [n] * n

        mono_stack = list()
        for i in range(n):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                right[mono_stack[-1]] = i
                mono_stack.pop()
            left[i] = mono_stack[-1] if mono_stack else -1
            mono_stack.append(i)
        
        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
        return ans
```

0610 更新 vivo提前批第二题（DP难起来总可以让人连题目都看不懂）

* 887. 鸡蛋掉落
方法一：动态规划 + 二分搜索
状态可以表示成 (K,N)，其中 K 为鸡蛋数，N 为楼层数
```python
class Solution:
    def superEggDrop(self, K: int, N: int) -> int:
        # 记忆化搜索，键为(k,n),值为测试次数
        self.memo = {}
        return self.dp(K, N)

    def dp(self,k, n):
        if (k, n) not in self.memo:
            if n == 0:
                ans = 0
            elif k == 1:
                ans = n
            else:
                lo, hi = 1, n
                # lo,li相差一或者相当，找到极值
                while lo + 1 < hi:
                    x = (lo + hi) // 2
                    t1 = self.dp(k-1, x-1)
                    t2 = self.dp(k, n-x)
                    # 最大的满足 T1(X)<T2(X)的X0​
                    if t1 < t2:
                        lo = x
                    elif t1 > t2:
                        hi = x
                    else:
                        lo = hi = x

                ans = 1 + min(max(self.dp(k-1, x-1), self.dp(k, n-x))
                                  for x in (lo, hi))

            self.memo[k, n] = ans
        return self.memo[k, n]
```
