# 2020中兴捧月算法大赛 阿尔法（MOT）赛道 Rank2
更多细节请查看我的博客：https://blog.csdn.net/weixin_42907473/article/details/106950555
### 1、环境配置
```
conda create -n fast python=3.7
cd  fast-master/
pip install -r requirments.txt
```
### 2、运行
```
cd fast-master/src/
source activate fast
python demo.py --data_root '测试数据路径' 
```
待代码运行结束，结果输出在fast-master/results/下

### 3、赛题分析
题目要求不能使用额外检测器，因此像FairMOT这种集成式算法就不再适用，CenterTrack尽管在MOTchallenge public赛道上表现亮眼，但其实现方式也是用到了自身的检测器（我本来想用这个模型的，结果发现其代码实现方式有点作弊，用到了自身检测结果），因此弃用。

相对而言，DeepSORT这种模型的拓展性就比较好，（测试只用数据关联部分+reid模型）但是经过测试，其显存占用和内存占用均超出限制，且执行速度很低（低于20FPS），对于比赛的评分标准十分不友好。

我认为在给定检测的条件下，MOTA通过reid的收益并不明显，主要原因是reid跨域的问题以及遮挡问题的存在，导致其可用性很低；因此其主要差距来源于对检测框的后处理，以及关联策略的选用，最后权衡之下，放弃了reid模型，这样即没有了显存的占用，减少了内存的占用，也提高了运行的速度。

### 4、算法思路
1)设定阈值先过滤一下给定的检测框，然后再通过nms来抑制重复框。

2)设计了轨迹评分函数：score = tracklet_len/max_len + track.score，在轨迹池的数量大于一定值时，优先匹配得分较高的轨迹。

3)计算卡尔曼预测的位置和检测之间的距离矩阵，并用Jonker-Volgenant 算法来进行分配。

4)计算未关联轨迹和未关联检测之间的IOU矩阵（内部值为1-iou），并用Jonker-Volgenant 算法来进行分配。

