
###                                                     决策树

#### 算法基本流程：

决策树的生成过程如下：

1. 选定最优划分属性，
2. 根据划分属性再进行子集递归

递归停下来的条件是：1）当前节点包含的样本全属于同一类别。

2）属性集为空，**或者所有样本在属性A上的值相同**（这个我真理解不了。。不明白为啥一个样本会有不同属性值），在这种情况下取样本最多的类别。称为当前结点的**后验分布**。

3）当前结点包含的样本集合为空，**在这种情况下取父结点样本最多的类别**（如果父结点的样本所属类别数一样咋办）。称为当前结点的**先验分布**。



#### 选定最优划分准则：

如何选择划分准则，使得到分支结点所包含的样本尽可能属于同一类别（纯度越来越高）。

第一种办法是采用**信息增益**：

###### 定义当前样本集合D中第k类样本所占比例为$P_{k}$, (k=1,2,3,,,,),则D的信息增益为：


$$
Ent(D)=-\sum^{y}_{k=1}p_{k}log_{2}P_{k}
$$
取值范围是0～$log_{2}n$,











