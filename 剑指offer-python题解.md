
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

## 3、数组中的重复数字
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
