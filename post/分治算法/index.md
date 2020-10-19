

​                                                  算法第一课：分治

​    分治的核心在于将复杂的问题递归的分为多个子问题求解（独立的，性质相同），先解决子问题，再把子问题合并起来解决原问题。

​    分治的一个典型的例子是归并排序，它的步骤如下：

1. 分解等待排序的 n个元素的序列成为n/2 个元素的两个子序列。
2. 使用归并排序递归排序两个子序列。
3. 合并两个已排序的子序列以产生已排序的答案。

我们来看下它的时间代价：在步骤1里分解 需要$log_2^N$步，步骤2的时间代价是O(n)，步骤3里的merge 需要C(n), 所以总的时间代价是： n*$log_2^N$ + C(n). 

可以用这个思路去分析其他的分治例子。

以下例子都用python实现。

题目一：求出一个数组中的逆序对。

# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
​        #用归并排序，归并拼接后用计算排序时元素的index变动了少
​        _,s=self.MergeSort(data)
​        return s%1000000007
​    def MergeSort(self,data):
​        n=len(data)
​        #递归基
​        if n==1:return data, 0
​        #分两半来排序
​        part1,part2=data[:n//2],data[n//2:]
​        sorted_part1,s1=self.MergeSort(part1)
​        sorted_part2,s2=self.MergeSort(part2)
​        #排序后拼接这两半，拼接后先计数，然后将两个有序序列合并
​        s,sorted_temp=0,sorted_part1+sorted_part2
​        #用p、q两个指针指向两段，计算q中每个元素离插入点的index差
​        p,q,len1,len_all=0,sorted_temp.index(sorted_part2[0]),len(sorted_part1),len(sorted_temp)
​        while p<len1 and q<len_all:
​            #移动p使p成为插入排序的插入点，计算要移动多少个位置
​            while p<len1:
​                if sorted_temp[q]<sorted_temp[p]:
​                    s+=len1-p
​                    break
​                p+=1
​            q+=1
​        #完成排序，并把排序后的内容回溯给上一级做准备
​        l=[]
​        p,q=0,sorted_temp.index(sorted_part2[0])
​        while p<len1 and q<len_all:
​            if sorted_temp[p]<sorted_temp[q]:
​                l.append(sorted_temp[p])
​                p+=1
​            else:
​                l.append(sorted_temp[q])
​                q+=1
​        if p==len1:l+=sorted_temp[q:]
​        if q==len_all:l+=sorted_part1[p:]
​        return l,s+s1+s2

题目二：已知一个数组中有正数也有负数，求出一个最大子数组。

其实这道题还可以用动态规划的方法，时间代价是O(n), 这里用的是分治的做法，时间代价跟上文一样。

int MaxSubSum(int *arr,int Left,int Right)
 2 {
 4     int MaxLeftSum,MaxRightSum;  //最大连续子数组没被中间的center切开时左右部分最大连续子数组之和
 6     int MaxLeftPartSum,MaxRightPartSum;//最大连续子数组被center切开时，最大子数组左右两部分的和
 8     int LeftPartSum,RightPartSum;  //临时变量，用于存储计算出来的和
 9     int Center,i;
10
11   
12     if(Left == Right)   //整个数组只有一个元素
13     {
14         if(arr[Left] > 0)  
15             return arr[Left];  
16         else  
17             return 0;
18     }
19
20     //递归调用。分别计算左右子数组的最大和子数组。
21     //即假设最大和子数组没有被Center切割
22     Center = (Left+Right)/2;  
23     MaxLeftSum = MaxSubSum(arr,Left,Center);  
24     MaxRightSum = MaxSubSum(arr,Center+1,Right);  
25
26     //假设最大和子数组被Center切开的情况
27     //那么需要从Center开始向两侧计算
28     MaxLeftPartSum = 0;  
29     LeftPartSum = 0;  
30     for(i = Center ; i >= Left; --i )   //从center向左边计算
31     {  
32         LeftPartSum += arr[i];  
33         if(LeftPartSum > MaxLeftPartSum)  
34             MaxLeftPartSum = LeftPartSum;  
35     }  
36     MaxRightPartSum = 0;  
37     RightPartSum = 0;  
38     for(i = Center+1 ; i <= Right ; ++i)  //从center向右计算
39     {  
40         RightPartSum += arr[i];  
41         if(RightPartSum > MaxRightPartSum)  
42             MaxRightPartSum = RightPartSum;  
43     }
44     //返回三者中的最大值。
45     return max(max(MaxLeftSum,MaxRightSum),MaxLeftPartSum+MaxRightPartSum);  



题目三：实现 `pow(x, n) `，即计算 `x` 的 `n` 次幂函数。

class Solution {
    public double myPow(double x, int n) {
        //long类型接收负数次方
        long N = n;
        if(n < 0){
            N = -n;
            x = 1 / x;
        }
        return fastPow(x, N);
    }
    //转换形参类型
    public double fastPow(double x, long n){
        if(n == 0){
            return 1;
        }
        double res = fastPow(x, n / 2);
        //每一次递归后都会先执行到return，再向下递归。
        if(n % 2 == 0){
            return res * res;
        }else{
            return res * res * x;
        } 
    }
}









