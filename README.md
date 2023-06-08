# data_process


1.运行脚本
python data_process.py

2. 文件结构
shapeNet数据的特征生成在 ./static/features 文件夹内
scanNet数据持久化、KDTree持久化、处理记录以及处理的结果都保存在 ./static 文件夹下

3.大致处理流程
通过PointNet++预训练模型的前面几层进行数据的特征提取
然后通过构建KDTree进行特征的相似性检索  在shpaeNet数据集中
的对应的类别中选择与scanNet特征最相似的点云数据进行配对
