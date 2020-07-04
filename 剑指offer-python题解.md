以下是本人基于剑指offer的书的思路，用Python写得代码，个人整理使用，较全，
建议配合牛客网OJ使用 https://www.nowcoder.com/ta/coding-interviews
sssssssssssssssssssssss
题目序号按照剑指offer第二版

## 第2章 面试需要的基本知识
* 3、数组中的重复数字
* 4、二维数组的查找
* 5、替换空格
* 6、从尾到头打印链表
* 7、重建二叉树
* 8、二叉树的下一个节点
* 9、用两个栈实现队列（栈、队列）
* 10、斐波那契数列（DP）
扩展 跳台阶、变态跳台阶
* 11、旋转数组的最小数字（二分）    
* 12、矩阵中的路径（DFS）
* 13、机器人的运动范围（BFS、回溯）
* 14、剪绳子（DP）
* 15、二进制中1的个数

## 第3章 高质量的代码
* 16、数值的整数次方
* 17、打印从1到最大的n位数
* 18、删除链表的节点
* 19、正则表达式的匹配
* 20、关于数值的字符串
* 21、调整数组顺序使奇数位于偶数前面
* 22、链表中倒数第k个节点
* 23、链表中环的入口节点
* 24、反转链表
* 25、合并两个排序的链表
* 26、树的子结构

## 第4章 解决面试题的思路
* 27、二叉树的镜像
* 28、对称的二叉树
* 29、顺时针打印矩阵
* 30、包含min函数的栈
* 31、栈的压入、弹出序列
* 32、从上到下打印树
* 33、二叉搜索树的后续遍历序列
* 34、二叉树和为某一值的路径
* 35、复杂链表的复制
* 36、二叉搜索树与双向链表
* 37、序列化二叉树
* 38、字符串的排列

## 第5章 优化空间和时间效率
* 39、数组中出现次数超过一半的数字
* 40、最小的k个数
* 41、数据流中的中位数
* 42、连续子数组的最大和
* 43、1~n整数中1出现的次数
* 44、数字序列中某一位的数字
* 45、把数组排成最小的数
* 46、把数字翻译成字符串
* 47、礼物的最大价值
* 48、最长不含重复字符的子字符串
* 49、丑数
* 50、第一个只出现一次的字符
* 51、数组中的逆序对
* 52、两个链表的第一个公共节点

## 第6章 面试中的各项能力
* 53、在排序数组中查找数字
* 54、二叉搜索树的第k大节点
* 55、二叉树的深度
* 56、数组中数字出现的次数
* 57、和为s的数字
* 58、翻转字符串
* 59、队列的最大值
* 60、n个骰子的点数
* 61、扑克牌的顺子
* 62、圆圈中最后剩下的数字
* 63、股票的最大收益
* 64、求1+2+。。。+n
* 65、不用加减乘除做加法
* 66、构造乘积数组

## 第7章 两个面试案例
* 67、把字符串转化为整数 
* 68、树中两个节点的最低公共祖先

## 3、数组中的重复数字（P39）

题目描述
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

思路：

(1) 从头到尾扫描 时间复杂度O(nlogn)

(2) 哈希表。每扫描到一个数字，判断是否存在在哈希表中，如果存在则找到，如果不存在则存入哈希表。时间复杂度O(n)，空间复杂度O(n)（需要维持一个哈希表）

(3) 数组中数字在0~n-1中，如果数组没有重复的，那么当数组排序后，数字i将出现在下标为i的位置。如果有重复的，那么某些位置会有多个数字，某些位置会没有数字
重排数组，当扫描到下标为i的数字时，首先比较这个数字（numbers[i]）是不是等于i,如果是，扫描下一个数字，如果不是，则将它与第numbers[i]个数字（值为numbers[numbers[i]]）比较，如果相等，则找到一个重复的数字，如果不想等，则把第i个数字与第numbers[i]个数字交换（numbers[i]放到下标为numbers[i]的位置，numbers[val]放到i的位置上，val = numbers[i],注意交换顺序），重复该过程，直到找到一个重复元素。
 ```python
 # -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        i = 0
        while i < len(numbers):
            val = numbers[i]
            if i != val:
                if numbers[numbers[i]] == val:
                    duplication[0] = val
                    return True
                else:
                    numbers[i] = numbers[val]
                    numbers[val] = val
            else:
                i += 1
        return False  
```

扩展：LeetCode 287. 寻找重复数（二分查找）

给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

要求：
不能更改原数组（假设数组是只读的）。
只能使用额外的 O(1) 的空间。
时间复杂度小于 O(n2) 。
数组中只有一个重复的数字，但它可能不止重复出现一次。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        n = len(nums) -1
        left = 1
        right = n
        while left < right:
            mid = left + (right - left) // 2
            cnt = 0
            for num in nums:
                if num <= mid:
                    cnt += 1
            # 根据抽屉原理，小于等于 4 的数的个数如果严格大于 4 个，
            # 此时重复元素一定出现在 [1, 4] 区间里
            if cnt > mid:
                # 重复的元素一定出现在 [left, mid] 区间里
                right = mid
            else:
                # if 分析正确了以后，else 搜索的区间就是 if 的反面
                # [mid + 1, right]
                left = mid + 1
        return left
  
```     
## 12、矩阵中的路径（p89）

题目描述
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

DFS/回溯

#遍历矩阵中的每一个位置

```python
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if not path:
            return True
        if not matrix:
            return False
           
        x = [list(matrix[cols*i:cols*i+cols]) for i in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if self.exist_helper(x, i, j, path):
                    return True
        return False
    
    def exist_helper(self, matrix, i, j, p):
        if matrix[i][j] == p[0]:
            if not p[1:]:
                return True
            matrix[i][j] = ''
            if i > 0 and self.exist_helper(matrix, i-1, j, p[1:]):
                return True
            if i < len(matrix)-1 and self.exist_helper(matrix, i+1, j ,p[1:]):
                return True
            if j > 0 and self.exist_helper(matrix, i, j-1, p[1:]):
                return True
            if j < len(matrix[0])-1 and self.exist_helper(matrix, i, j+1, p[1:]):
                return True
            matrix[i][j] = p[0]
            return False
        else:
            return False
```

## 13、机器人的运动范围（P92）

题目描述
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

思路：从[0,0]开始BFS
```
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        if threshold <0:
            return 0
        return self.dfs(threshold, rows, cols)
        
    def dfs(self,threshold,rows,cols):
        visited = [[0]*cols for _ in range(rows)]
        visited[0][0] = 1
        from collections import deque
        queue = deque([[0,0]])
        count = 1
        while queue:
            [x,y] = queue.popleft()
            for [x_new,y_new] in [[x, y + 1],[x + 1, y]]: # [x, y - 1],  [x - 1, y]，
                if self.get_sum(x_new,y_new) > threshold:
                    continue
                if not 0<=x_new<rows or not 0<=y_new<cols:
                    continue
                if visited[x_new][y_new] == 0:
                    visited[x_new][y_new] = 1    
                    queue.append([x_new,y_new])
                    count+=1            
        return count
    
    def get_sum(self,a,b):
        ans = 0
        while a != 0:
            ans += a % 10
            a //= 10
        while b != 0:
            ans += b % 10
            b //= 10
        return ans
```
## 14、剪绳子（DP）
DP的解法
```
# -*- coding:utf-8 -*-
class Solution:
    def cutRope(self, number):
        # write code here
        if number == 2:
            return 1
        if number == 3:
            return 2
        memo = [0,1, 2, 3] + [0] * (number - 3)
        for i in range(4,number+1):
            for j in range(1,i//2+1):
                memo[i] = max(memo[i],memo[j]*memo[i-j])
        return memo[-1]
```
## 15、二进制中1的个数
一个很巧妙的解法
```
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        
        count = 0
        if n < 0:
            n = n & 0xffffffff
        while n:
            count += 1
            # n-1:最右边的一个1开始的所有位都取反了
            # (n - 1) & n 相当于每次减掉最右边的一个1
            n = (n - 1) & n
        return count
```
