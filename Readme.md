# EfficientGCN_paddle
## 1.简介
This is an unofficial code based on PaddlePaddle of IEEE 2022 paper:

[EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9729609)




![image](https://user-images.githubusercontent.com/59130750/171786592-c1d7d67c-7d0c-4816-b8ff-7f7a5fe0320b.png)

这是一篇骨骼点动作识别领域的文章，文章提出了EfficientGCN模型，该模型在MIB网络中结合可分离的卷积层，利用图卷积网络对视频动作进行识别，骨骼点数据相对于传统RGB数据更具解释性与鲁棒性。该方法相较于传统中参数量较大的双流特征提取方式，在模型的前端选择融合三个输入分支并输入主流模型提取特征，通过这种方式减小了模型的复杂度。  
论文地址：[EfficientGCN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9729609)  
原论文代码地址：[EfficientGCN Code](https://gitee.com/yfsong0709/EfficientGCNv1)  

## 2.复现精度

注：NTU RGB+D 60数据集,EfficientGCN-B0模型下的x-sub和x-view分别对应2001和2002模型

 NTU RGB+D 60数据集，EfficientGCN-B0模型  | X-sub（2001）  | X-view （2002） 
 ---- | ----- | ------  
 Paper  | 90.2% | 94.9% 
 Paddle  | 90.2% | 94.99%  
 
 在NTU RGB+D 60数据集上基本达到验收标准  
 训练日志和模型权重：  https://github.com/small-whirlwind/EfficientGCN_paddle/tree/main/workdir_pad

 aistudio的实现方式：  
 在./tasks/文件夹内运行CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 0 -c 2001 -e  
 在./tasks/文件夹内运行CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 0 -c 2002 -e  
 选择要测试的模型即可  
 
 ## 3.环境依赖
- 硬件：GeForce RTX 2080 Ti  
- Based on Python3 (anaconda, >= 3.5) and PyTorch (>= 1.6.0).
- paddlePaddle-gpu==2.2.2  
- padddlenlp==2.2.6
- `pip install -r requirements.txt`  
## 4.数据集和预训练模型下载
复现任务是在NTU RGB+D 60数据集上进行的，只需要骨骼点1-17的部分，可以从这里下载https://drive.google.com/file/d/1CUZnBtYwifVXS21yVg62T-vrPVayso5H/view  

预训练模型，在这里下载https://drive.google.com/drive/folders/1HpvkKyfmmOCzuJXemtDxQCgGGQmWMvj4 。在本次任务中，下载2001,2002即可。

但此处但此处下载的ckpy文件适配于pytorch框架，在此给出两种解决方案：  
- 直接使用项目pretrained文件夹中转换好的ckpy  
- 通过本项目中的transferForPth.py文件进行模型转换，将.pth文件转换为适配paddle的.pdparams文件。  
## 5.数据预处理
### 5.1 config文件生成
输入数据集路径、预处理后的数据集存放路径、预训练模型路径等，生成config文件  
```
python scripts/modify_configs.py --root_folder <path/to/save/numpy/data> --ntu60_path <path/to/ntu60/dataset> --ntu120_path <path/to/ntu120/dataset> --pretrained_path <path/to/save/pretraiined/model> --work_dir <path/to/work/dir>
```  
示例：
```
python3 scripts/modify_configs.py --root_folder /share/liukaiyuan/NTU60/paddle_xyf/data/npy_dataset/ --ntu60-path /share/liukaiyuan/NTU60/paddle_xyf/nturgbd_skeletons_s001_to_s017 --ntu120_path /share/NTU-RGB-D120 --pretrained_path /home/liukaiyuan/xyf/EfficientGCN_torch/pretrained  --workdir /share/liukaiyuan/NTU60/paddle_xyf/workdir_pad
```

### 5.2 数据预处理

```
python main.py -c 2001 -gd -np  
python main.py -c 2002 -gd -np
```

最终，经过预处理后的数据集文件夹格式如下所示：
```
-dataset -xview
        |-xsub
        |-data/npy_datatset -transformed
                           |-original
        |-nturgbd_skeletons_s001_to_s017
        |-workdir_pad

```


## 5 模型训练
在终端输入如下命令行进行训练：
```
python main.py -c <config>
```
在本次复现项目中，针对2001,2002两个model，输入

x-sub(2001)
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 0 -c 2001
```

x-view(2002)
```
CUDA_VISIBLE_DEVICES=1 python3 main.py --gpus 0 -c 2002
```

部分训练输出如下：

```
Loss: 0.0024, LR: 0.0001: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2505/2505 [11:41<00:00,  3.57it/s]
[ 2022-06-02 13:32:03,791 ] Epoch: 69/70, Training accuracy: 39914/40080(99.59%), Training time: 701.38s
[ 2022-06-02 13:32:03,792 ] 
[ 2022-06-02 13:32:03,793 ] Evaluating for epoch 69/70 ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1031/1031 [01:53<00:00,  9.09it/s]
[ 2022-06-02 13:33:57,190 ] Top-1 accuracy: 14762/16487(89.54%), Top-5 accuracy: 16194/16487(98.22%), Mean loss:0.4057
[ 2022-06-02 13:33:57,190 ] Evaluating time: 113.39s, Speed: 145.48 sequnces/(second*GPU)
[ 2022-06-02 13:33:57,190 ] 
[ 2022-06-02 13:33:57,247 ] Saving model for epoch 69/70 ...
[ 2022-06-02 13:33:57,290 ] Best top-1 accuracy: 89.89%, Total time: 00d-13h-57m-19s
[ 2022-06-02 13:33:57,290 ] 
Loss: 0.0052, LR: 0.0000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2505/2505 [11:25<00:00,  3.65it/s]
[ 2022-06-02 13:45:22,672 ] Epoch: 70/70, Training accuracy: 39922/40080(99.61%), Training time: 685.38s
[ 2022-06-02 13:45:22,672 ] 
[ 2022-06-02 13:45:22,674 ] Evaluating for epoch 70/70 ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1031/1031 [02:03<00:00,  8.37it/s]
[ 2022-06-02 13:47:25,924 ] Top-1 accuracy: 14735/16487(89.37%), Top-5 accuracy: 16203/16487(98.28%), Mean loss:0.3986
[ 2022-06-02 13:47:25,924 ] Evaluating time: 123.25s, Speed: 133.85 sequnces/(second*GPU)
[ 2022-06-02 13:47:25,925 ] 
[ 2022-06-02 13:47:25,988 ] Saving model for epoch 70/70 ...
[ 2022-06-02 13:47:26,025 ] Best top-1 accuracy: 89.89%, Total time: 00d-14h-10m-48s
[ 2022-06-02 13:47:26,026 ] 
[ 2022-06-02 13:47:26,026 ] Finish training!
```


## 6 模型测试

在终端输入如下命令行进行训练：
```
python main.py -c <config> -e
```
在本次复现项目中，针对2001,2002两个model，输入

x-sub(2001)
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 0 -c 2001 -e
```
注意，输入以上命令后需要选择测试的模型，作者训练好的达标模型标注为1号，输入数字1+回车即可

结果如下所示：
```
[ 2022-06-05 23:53:25,024 ] Saving folder path: /share/liukaiyuan/NTU60/paddle_xyf/workdir_pad/temp
[ 2022-06-05 23:53:25,024 ] 
[ 2022-06-05 23:53:25,025 ] Starting preparing ...
[ 2022-06-05 23:53:25,025 ] Saving model name: 2001_EfficientGCN-B0_ntu-xsub
[ 2022-06-05 23:53:25,037 ] GPU-0 used: 3.0MB
[ 2022-06-05 23:53:25,055 ] Dataset: ntu-xsub
[ 2022-06-05 23:53:25,055 ] Batch size: train-16, eval-16
[ 2022-06-05 23:53:25,055 ] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
[ 2022-06-05 23:53:25,055 ] Number of action classes: 60
[ 2022-06-05 23:53:28,235 ] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
[ 2022-06-05 23:53:28,269 ] Pretrained model: /home/liukaiyuan/xyf/EfGCN/pretrained/2001_EfficientGCN-B0_ntu-xsub.pdparams.tar
[ 2022-06-05 23:53:28,269 ] LR_Scheduler: cosine {'max_epoch': 70, 'warm_up': 10}
[ 2022-06-05 23:53:28,270 ] Optimizer: SGD {'momentum': 0.9, 'weight_decay': 0.0001, 'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7f6ef0ba0290>, 'use_nesterov': True}
[ 2022-06-05 23:53:28,270 ] Loss function: CrossEntropyLoss
[ 2022-06-05 23:53:28,270 ] Successful!
[ 2022-06-05 23:53:28,270 ] 
[ 2022-06-05 23:53:28,271 ] Loading evaluating model ...
[ 2022-06-05 23:53:28,272 ] Please choose the evaluating model from the following models.
[ 2022-06-05 23:53:28,272 ] Default is the initial or pretrained model.
[ 2022-06-05 23:53:28,273 ] (31) accuracy: 89.89% | training time: 2022-06-01 23-36-34
[ 2022-06-05 23:53:28,273 ] (32) accuracy: 81.51% | training time: 2022-06-04 00-04-26
[ 2022-06-05 23:53:28,273 ] (33) accuracy: 88.22% | training time: 2022-04-27 02-06-45
[ 2022-06-05 23:53:28,273 ] (44) accuracy: 89.34% | training time: 2022-04-29 23-33-28
[ 2022-06-05 23:53:28,273 ] (45) accuracy: 84.33% | training time: 2022-05-31 19-21-21
[ 2022-06-05 23:53:28,273 ] (52) accuracy: 89.77% | training time: 2022-06-01 22-58-19
[ 2022-06-05 23:53:28,273 ] Your choice (number of the model, q for quit): 
[ 2022-06-05 23:53:28,273 ] 31
[ 2022-06-05 23:54:08,988 ] Successful!
[ 2022-06-05 23:54:08,988 ] 
[ 2022-06-05 23:54:08,988 ] Starting evaluating ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1031/1031 [01:48<00:00,  9.53it/s]
[ 2022-06-05 23:55:57,127 ] Top-1 accuracy: 14820/16487(89.89%), Top-5 accuracy: 16223/16487(98.40%), Mean loss:0.3859
[ 2022-06-05 23:55:57,127 ] Evaluating time: 108.14s, Speed: 152.55 sequnces/(second*GPU)
[ 2022-06-05 23:55:57,127 ] 
[ 2022-06-05 23:55:57,143 ] Finish evaluating!
```


x-view(2002)
```
CUDA_VISIBLE_DEVICES=1 python3 main.py --gpus 0 -c 2002 -e
```
同理，输入以上命令后需要选择测试的模型，作者训练好的达标模型标注为1号，输入数字1+回车即可
部分测试输出如下：

```
[ 2022-06-06 00:32:03,271 ] Saving folder path: /share/liukaiyuan/NTU60/paddle_xyf/workdir_pad/temp
[ 2022-06-06 00:32:03,272 ] 
[ 2022-06-06 00:32:03,272 ] Starting preparing ...
[ 2022-06-06 00:32:03,272 ] Saving model name: 2002_EfficientGCN-B0_ntu-xview
[ 2022-06-06 00:32:03,314 ] GPU-0 used: 6460.0MB
[ 2022-06-06 00:32:03,316 ] 
[ 2022-06-06 00:32:03,316 ] GPU-0 is occupied!
[ 2022-06-06 00:32:03,338 ] Dataset: ntu-xview
[ 2022-06-06 00:32:03,338 ] Batch size: train-16, eval-16
[ 2022-06-06 00:32:03,338 ] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
[ 2022-06-06 00:32:03,338 ] Number of action classes: 60
[ 2022-06-06 00:32:06,812 ] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
[ 2022-06-06 00:32:06,841 ] Pretrained model: /home/liukaiyuan/xyf/EfGCN/pretrained/2002_EfficientGCN-B0_ntu-xview.pdparams.tar
[ 2022-06-06 00:32:06,842 ] LR_Scheduler: cosine {'max_epoch': 70, 'warm_up': 10}
[ 2022-06-06 00:32:06,843 ] Optimizer: SGD {'momentum': 0.9, 'weight_decay': 0.0001, 'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7fbdc04534d0>, 'use_nesterov': True}
[ 2022-06-06 00:32:06,843 ] Loss function: CrossEntropyLoss
[ 2022-06-06 00:32:06,843 ] Successful!
[ 2022-06-06 00:32:06,843 ] 
[ 2022-06-06 00:32:06,843 ] Loading evaluating model ...
[ 2022-06-06 00:32:06,844 ] Please choose the evaluating model from the following models.
[ 2022-06-06 00:32:06,844 ] Default is the initial or pretrained model.
[ 2022-06-06 00:32:06,844 ] (2) accuracy: 94.03% | training time: 2022-04-29 23-50-44
[ 2022-06-06 00:32:06,844 ] (3) accuracy: 94.78% | training time: 2022-06-01 23-37-31
[ 2022-06-06 00:32:06,844 ] (6) accuracy: 94.22% | training time: 2022-05-31 03-29-21
[ 2022-06-06 00:32:06,844 ] (9) accuracy: 90.68% | training time: 2022-05-31 19-21-24
[ 2022-06-06 00:32:06,844 ] (10) accuracy: 94.69% | training time: 2022-06-02 17-38-01
[ 2022-06-06 00:32:06,844 ] (11) accuracy: 94.18% | training time: 2022-05-30 06-26-24
[ 2022-06-06 00:32:06,844 ] (12) accuracy: 94.10% | training time: 2022-06-01 05-11-02
[ 2022-06-06 00:32:06,844 ] Your choice (number of the model, q for quit): 
[ 2022-06-06 00:32:06,844 ] 3
[ 2022-06-06 00:32:27,550 ] Successful!
[ 2022-06-06 00:32:27,550 ] 
[ 2022-06-06 00:32:27,550 ] Starting evaluating ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1184/1184 [02:03<00:00,  9.57it/s]
[ 2022-06-06 00:34:31,248 ] Top-1 accuracy: 17944/18932(94.78%), Top-5 accuracy: 18798/18932(99.29%), Mean loss:0.1925
[ 2022-06-06 00:34:31,249 ] Evaluating time: 123.70s, Speed: 153.15 sequnces/(second*GPU)
[ 2022-06-06 00:34:31,249 ] 
[ 2022-06-06 00:34:31,266 ] Finish evaluating!

```




## 7 附录

 信息| 描述
 ---- | ----- 
 作者  | small-whirlwind 
 日期  | 2022年6月
 框架版本 | PaddlePaddle-gpu==2.2.0
 应用场景 | 骨架动作识别
 硬件支持 | GPU
 Aistudio |[Efficient_paddle](https://aistudio.baidu.com/aistudio/projectdetail/4167040?_=1655386061207&contributionType=1)
 
 感谢百度飞桨团队提供的技术支持！
