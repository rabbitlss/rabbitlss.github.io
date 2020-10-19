
做了机器学习下的文本分类探索，接下来是深度学习的一些算法探索。

**现在文本表示方法的缺陷**：

One-hot / Bag of Words：转换得到的向量维度很高，

TF-IDF：没有考虑单词与单词之间的关系，只是进行了统计。

N-gram: 考虑到了单词与单词之间的关系，但是转换得到的响亮维度很高。

**与传统机器学习不同，深度学习既提供特征提取功能，也可以完成分类的功能**。

深度学习也可以用于文本表示，还可以将其映射到一个低纬空间。其中比较典型的例子有：FastText、Word2Vec和Bert。在本章我们将介绍FastText，将在后面的内容介绍Word2Vec和Bert。

Fasttext是Facebook开源的一个文本分类包，是一种典型的深度学习词向量的表示方法，它非常简单通过Embedding层将单词映射到稠密空间，然后将句子中所有的单词在Embedding空间中进行平均，进而完成分类操作。

所以FastText是一个三层的神经网络，输入层、隐含层和输出层

![屏幕快照 2020-07-28 上午8.34.39](/Users/lishanshan/Desktop/屏幕快照 2020-07-28 上午8.34.39.png)



这是keras实现的FastText网络结构：

![屏幕快照 2020-07-28 上午8.35.01](/Users/lishanshan/Desktop/屏幕快照 2020-07-28 上午8.35.01.png![屏幕快照 2020-07-28 上午8.35.01](/Users/lishanshan/Desktop/屏幕快照 2020-07-28 上午8.35.01.png)

FastText在文本分类任务上，是优于TF-IDF的：

- FastText用单词的Embedding叠加获得的文档向量，将相似的句子分为一类
- FastText学习到的Embedding空间维度比较低，可以快速进行训练

**参考以下深度学习的论文：**

Bag of Tricks for Efficient Text Classification, https://arxiv.org/abs/1607.01759

Enriching Word Vectors with Subword Information , https://arxiv.org/pdf/1607.04606.pdf



#### 基于FastText的文本分类：

```
import pandas as pd
from sklearn.metrics import f1_score
import fasttext

# 转换为FastText需要的格式
path_data='/Users/lishanshan/Workspace/Datawhale/NLP/train_set.csv'
train_df = pd.read_csv(path_data, sep='\t', nrows=15000)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=3,
                                  verbose=2, minCount=1, epoch=30, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
```

----0.8235



优化方法：

1. 调整 wordNgrams=2 -> 3, 数字越大精度越准确。
2. epoch=20 -> 30 , 训练次数越多精度越高。



