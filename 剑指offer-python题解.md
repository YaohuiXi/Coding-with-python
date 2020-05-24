以下是本人基于剑指offer的书的思路，用Python写得代码，个人整理使用，较全，
建议配合牛客网OJ使用 https://www.nowcoder.com/ta/coding-interviews

题目序号按照剑指offer第二版
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

## 3、数组中的重复数字（P39）

题目描述
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

思路：
(1)从头到尾扫描 时间复杂度O(nlogn)
(2)哈希表。每扫描到一个数字，判断是否存在在哈希表中，如果存在则找到，如果不存在则存入哈希表。时间复杂度O(n)，空间复杂度O(n)（需要维持一个哈希表）
(3)数组中数字在0~n-1中，如果数组没有重复的，那么当数组排序后，数字i将出现在下标为i的位置。如果有重复的，那么某些位置会有多个数字，某些位置会没有数字
重排数组，当扫描到下标为i的数字时，首先比较这个数字（numbers[i]）是不是等于i,如果是，扫描下一个数字，如果不是，则将它与第numbers[i]个数字（值为numbers[numbers[i]]）比较，如果相等，则找到一个重复的数字，如果不想等，则把第i个数字与第numbers[i]个数字交换（numbers[i]放到下标为numbers[i]的位置，numbers[val]放到i的位置上，val = numbers[i],注意交换顺序），重复该过程，直到找到一个重复元素。
 ```
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
