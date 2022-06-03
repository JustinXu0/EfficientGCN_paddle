# EfficientGCN_paddle
## 1.简介
This is an unofficial code based on PaddlePaddle of IEEE 2022 paper:

[EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9729609)




![image](https://user-images.githubusercontent.com/59130750/171786592-c1d7d67c-7d0c-4816-b8ff-7f7a5fe0320b.png)

这是一篇骨骼点动作识别领域的文章，文章提出了EfficientGCN模型，该模型在MIB网络中结合可分离的卷积层，利用图卷积网络对视频动作进行识别，骨骼点数据相对于传统RGB数据更具解释性与鲁棒性。该方法相较于传统中参数量较大的双流特征提取方式，在模型的前端选择融合三个输入分支并输入主流模型提取特征，通过这种方式减小了模型的复杂度。  
论文地址：[EfficientGCN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9729609)  
原论文代码地址：[EfficientGCN Code](https://gitee.com/yfsong0709/EfficientGCNv1)  

## 2.复现精度


 NTU RGB+D 60数据集，EfficientGCN-B0模型  | X-sub（2001）  | X-view （2002） 
 ---- | ----- | ------  
 Paper  | 90.2% | 94.9% 
 Paddle  | 89.89% | 94.78%  
 
 在NTU RGB+D 60数据集上基本达到验收标准  
 训练日志：  
 模型权重：  
## 3.环境依赖
- 硬件：GeForce RTX 2080 Ti  
- Based on Python3 (anaconda, >= 3.5) and PyTorch (>= 1.6.0).
- paddlePaddle-gpu==2.2.2  
- padddlenlp==2.2.6
- `pip install -r requirements.txt`  
## 4.数据集
复现任务是在NTU RGB+D 60数据集上进行的，只需要骨骼点1-17的部分，可以从这里下载https://drive.google.com/file/d/1CUZnBtYwifVXS21yVg62T-vrPVayso5H/view  
## 5.数据预处理
### 5.1 config文件生成
输入数据集路径、预处理后的数据集存放路径、预训练模型路径等，生成config文件  
`python scripts/modify_configs.py --path <path/to/save/numpy/data> --ntu60_path <path/to/ntu60/dataset> --ntu120_path <path/to/ntu120/dataset> --pretrained_path <path/to/save/pretraiined/model> --work_dir <path/to/work/dir>`















