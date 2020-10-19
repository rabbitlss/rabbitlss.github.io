

常用的压缩文件：gz, 

在hdfs下不解压看文件的方式：hadoop fs -text  test.gz

在本地路径下：less test.gz

判断一个文件是否为空，用stat：

stat test.gz 

gz文件会有头部信息，因此最小的size是28 bytes. 如果 size=0，可以尝试预览下文件，看是否缺了头部信息。如果是的话就为坏文件。

压缩文件的几种算法：

