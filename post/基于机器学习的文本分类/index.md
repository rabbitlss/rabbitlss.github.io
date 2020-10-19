

 这节讲的是用机器学习的方法对文本进行归类，

#### 常用的文本处理方法和机器学习方法

TF-IDF是处理文本离散值的一种方式，

 sklearn中有两种实现包：

TfidfVectorizer： 把一组文档转换成TF-IDF值的矩阵。

CountVectorizer: 把一组文档转换成TF-IDF的标准化的n-gram （考虑到单个词，和词语之间的顺序）矩阵。

主要的文本表示方法有：

1. ##### One-hot: 

   这里的One-hot与数据挖掘任务中的操作是一致的，即将每一个单词使用一个离散的向量表示。具体将每个字/词编码一个索引，然后根据索引进行赋值.

   ```python
   句子1：我 爱 北 京 天 安 门
   句子2：我 喜 欢 上 海
   ```

   首先对所有句子的字进行索引，即将每个字确定一个编号：

   ```python
   {
       '我': 1, '爱': 2, '北': 3, '京': 4, '天': 5,
     '安': 6, '门': 7, '喜': 8, '欢': 9, '上': 10, '海': 11
   }
   ```

   在这里共包括11个字，因此每个字可以转换为一个11维度稀疏向量：

   ```
   我：[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   爱：[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ...
   海：[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
   ```

2. ##### Bag of Words

在sklearn中可以直接`CountVectorizer`来实现这一步骤: 

```
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
vectorizer.fit_transform(corpus).toarray()
```

3. ##### N-gram:

   N-gram与Count Vectors类似，不过加入了相邻单词组合成为新的单词，并进行计数。这样做的好处是避免有完全相同的组合（文章由文字组成）带来完全相同的表示。

4. ##### TF-IDF

TF-IDF 分数由两部分组成：第一部分是**词语频率**（Term Frequency），第二部分是**逆文档频率**（Inverse Document Frequency）。其中计算语料库中文档总数除以含有该词语的文档数量，然后再取对数就是逆文档频率。

```
TF(t)= 该词语在当前文档出现的次数 / 当前文档中词语的总数
IDF(t)= log_e（文档总数 / 出现该词语的文档总数）
```



目前用来做离散值分类的机器学习方法有：

1. ##### 基本模型

1）LinearRegression()：线性回归

2）LogisticRegression：逻辑回归，可以做多分类（OvR）

3）LogisticRegressionCV(Cs=10,penalty='l2',solver='lbfgs')：多C分类，CV（Cs），可以做多分类

2. ##### Lasso回归（分类）--L1正则化

Lasso(alpha=0.1) ：加L1正则项回归，不能同时拟合多个

LassoCV(n_alphas=100, alphas=None)：多个alpha，不能同时拟合多个

MultiTaskLasso(alpha=1.0) ：同上，但是只能同时拟合多个

MultiTaskLassoCV(n_alphas=100, alphas=None)：同上，但是只能同时拟合多个

3. ##### 感知器，仅使用错误样本更新模型

Perceptron(penalty=None, alpha=0.0001)：分类，参数penalty可选L1、L2和None，可以做二分类和多分类

4. ##### 岭回归（分类）--L2正则化

Ridge(alpha=1.0,solver="auto")：回归，参数含义前面已有，可以同时拟合多个

RidgeCV(alphas=(0.1, 1.0, 10.0))：回归，参数含义前面已有，可以同时拟合多个

RidgeClassifier(alpha=1.0,solver="auto")：分类，参数含义已知，可以做二分类和多分类

RidgeClassifierCV(alphas=(0.1, 1.0, 10.0))：分类，参数含义已知，可以做二分类和多分类

##### 5. 随机梯度下降SGD，在样本量（和特征数）很大时适用

SGDClassifier(loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15)：

分类，损失函数常用hinge和log—分别对应支持向量机SVM和logistic，可以做二分类和多分类

SGDClassifier(loss="squared_loss", penalty='l2', alpha=0.0001, l1_ratio=0.15)：

回归，可以同时拟合多个

6. ##### 贝叶斯岭回归（以前只知道贝叶斯分类）

BayesianRidge(alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6)：

回归，不能同时拟合多个



#### 基于机器学习的文本分类

1. 接下来我们将对比不同文本表示算法的精度，通过本地构建验证集计算F1得分。

```
# Count Vectors + RidgeClassifier
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

path_data='/Users/lishanshan/Workspace/Datawhale/NLP/train_set.csv'

# Count Vectors + RidgeClassifier
train_df = pd.read_csv(path_data, sep='\t', nrows=15000)
vectorizer = CountVectorizer(max_features=3000,stop_words={'3750','900','648'},ngram_range=(1,2))
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
```

----0.6205

精度比较低。

##### 基于TF-IDF做文本值处理，岭回归做文本分类。

2. ```
   import pandas as pd
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.linear_model import RidgeClassifier
   from sklearn.metrics import f1_score
   from sklearn.feature_extraction.text import TfidfVectorizer
   
   
   # TF-IDF +  RidgeClassifier
   
   path_data='/Users/lishanshan/Workspace/Datawhale/NLP/train_set.csv'
   train_df = pd.read_csv(path_data, sep='\t', nrows=15000)
   
   tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000, smooth_idf=False)
   train_test = tfidf.fit_transform(train_df['text'])
   print(train_test)
   
   clf = RidgeClassifier()
   clf.fit(train_test[:10000], train_df['label'].values[:10000])
   val_pred = clf.predict(train_test[10000:])
   print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
   ```

----0.8205

精度比较高

##### 他也是基于TF-IDF做文本值处理，岭回归做文本分类。

其中做了参数的优化：

1. ngram_range=(1,3) 调成 ngram_range=(1,2)

2. 加入停词：stop_words={'3750','900','648'}，从上一节数据分析中判断出这三个词有可能是标点符号。
3. smooth_idf=True - > smooth_idf=False
