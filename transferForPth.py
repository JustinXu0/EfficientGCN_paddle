import numpy as np
import torch
import paddle
import os


def transfer():
    # input_fp = "../pretrained/2014_EfficientGCN-B0_ntu-xview.pth.tar"
    # output_fp = "/home/liukaiyuan/xyf/EfGCN/2014_EfficientGCN-B0_ntu-xview.pdparams.tar"
    input_fp = "../pretrained/2013_EfficientGCN-B0_ntu-xsub.pth.tar"
    output_fp = "/home/liukaiyuan/xyf/EfGCN/pretrained/z2013_EfficientGCN-B0_ntu-xsub.pdparams.tar"
    # input_fp = "../pretrained/2002_EfficientGCN-B0_ntu-xview.pth.tar"
    # output_fp = "/home/liukaiyuan/xyf/EfGCN/pretrained/2002_EfficientGCN-B0_ntu-xview.pdparams.tar"
    # input_fp = "../pretrained/2001_EfficientGCN-B0_ntu-xsub.pth.tar"
    # output_fp = "/home/liukaiyuan/xyf/EfGCN/pretrained/2001_EfficientGCN-B0_ntu-xsub.pdparams.tar"

    torch_all_dict = torch.load(input_fp, map_location=torch.device('cpu'))
    torch_dict = torch_all_dict['model']
    paddle_dict = {}
   
    fc_names = ['input_branches.0.init_bn.weight', 
'input_branches.0.init_bn.bias', 
'input_branches.0.init_bn.running_mean', 
'input_branches.0.init_bn.running_var', 
'input_branches.0.init_bn.num_batches_tracked', 

'input_branches.0.stem_scn.conv.A', 
'input_branches.0.stem_scn.conv.edge', 

'input_branches.0.stem_scn.conv.gcn.weight', 
'input_branches.0.stem_scn.conv.gcn.bias', 

'input_branches.0.stem_scn.bn.weight', 
'input_branches.0.stem_scn.bn.bias', 
'input_branches.0.stem_scn.bn.running_mean', 
'input_branches.0.stem_scn.bn.running_var', 
'input_branches.0.stem_scn.bn.num_batches_tracked', 

'input_branches.0.stem_scn.residual.0.weight', 
'input_branches.0.stem_scn.residual.0.bias',

'input_branches.0.stem_scn.residual.1.weight', 
'input_branches.0.stem_scn.residual.1.bias', 
'input_branches.0.stem_scn.residual.1.running_mean', 
'input_branches.0.stem_scn.residual.1.running_var', 
'input_branches.0.stem_scn.residual.1.num_batches_tracked',

'input_branches.0.stem_tcn.conv.weight', 
'input_branches.0.stem_tcn.conv.bias', 

'input_branches.0.stem_tcn.bn.weight',
'input_branches.0.stem_tcn.bn.bias',
'input_branches.0.stem_tcn.bn.running_mean', 
'input_branches.0.stem_tcn.bn.running_var', 
'input_branches.0.stem_tcn.bn.num_batches_tracked', 

'input_branches.0.block-0_scn.conv.A', 
'input_branches.0.block-0_scn.conv.edge', 

'input_branches.0.block-0_scn.conv.gcn.weight', 
'input_branches.0.block-0_scn.conv.gcn.bias', 

'input_branches.0.block-0_scn.bn.weight', 
'input_branches.0.block-0_scn.bn.bias', 
'input_branches.0.block-0_scn.bn.running_mean', 
'input_branches.0.block-0_scn.bn.running_var', 
'input_branches.0.block-0_scn.bn.num_batches_tracked', 

'input_branches.0.block-0_scn.residual.0.weight', 
'input_branches.0.block-0_scn.residual.0.bias', 

'input_branches.0.block-0_scn.residual.1.weight', 
'input_branches.0.block-0_scn.residual.1.bias', 
'input_branches.0.block-0_scn.residual.1.running_mean', 
'input_branches.0.block-0_scn.residual.1.running_var', 
'input_branches.0.block-0_scn.residual.1.num_batches_tracked', 

'input_branches.0.block-0_att.att.fcn.0.weight', 
'input_branches.0.block-0_att.att.fcn.0.bias',

'input_branches.0.block-0_att.att.fcn.1.weight', 
'input_branches.0.block-0_att.att.fcn.1.bias', 
'input_branches.0.block-0_att.att.fcn.1.running_mean', 
'input_branches.0.block-0_att.att.fcn.1.running_var', 
'input_branches.0.block-0_att.att.fcn.1.num_batches_tracked', 

'input_branches.0.block-0_att.att.conv_t.weight', 
'input_branches.0.block-0_att.att.conv_t.bias', 

'input_branches.0.block-0_att.att.conv_v.weight', 
'input_branches.0.block-0_att.att.conv_v.bias',

'input_branches.0.block-0_att.bn.weight', 
'input_branches.0.block-0_att.bn.bias', 
'input_branches.0.block-0_att.bn.running_mean', 
'input_branches.0.block-0_att.bn.running_var', 
'input_branches.0.block-0_att.bn.num_batches_tracked', 

'input_branches.0.block-1_scn.conv.A', 
'input_branches.0.block-1_scn.conv.edge', 

'input_branches.0.block-1_scn.conv.gcn.weight', 
'input_branches.0.block-1_scn.conv.gcn.bias', 

'input_branches.0.block-1_scn.bn.weight', 
'input_branches.0.block-1_scn.bn.bias', 
'input_branches.0.block-1_scn.bn.running_mean', 
'input_branches.0.block-1_scn.bn.running_var',
'input_branches.0.block-1_scn.bn.num_batches_tracked', 

'input_branches.0.block-1_scn.residual.0.weight', 
'input_branches.0.block-1_scn.residual.0.bias',

'input_branches.0.block-1_scn.residual.1.weight', 
'input_branches.0.block-1_scn.residual.1.bias', 
'input_branches.0.block-1_scn.residual.1.running_mean',
'input_branches.0.block-1_scn.residual.1.running_var', 
'input_branches.0.block-1_scn.residual.1.num_batches_tracked', 

'input_branches.0.block-1_att.att.fcn.0.weight', 
'input_branches.0.block-1_att.att.fcn.0.bias', 

'input_branches.0.block-1_att.att.fcn.1.weight', 
'input_branches.0.block-1_att.att.fcn.1.bias',
'input_branches.0.block-1_att.att.fcn.1.running_mean', 
'input_branches.0.block-1_att.att.fcn.1.running_var', 
'input_branches.0.block-1_att.att.fcn.1.num_batches_tracked', 

'input_branches.0.block-1_att.att.conv_t.weight', 
'input_branches.0.block-1_att.att.conv_t.bias', 

'input_branches.0.block-1_att.att.conv_v.weight', 
'input_branches.0.block-1_att.att.conv_v.bias', 

'input_branches.0.block-1_att.bn.weight', 
'input_branches.0.block-1_att.bn.bias', 
'input_branches.0.block-1_att.bn.running_mean', 
'input_branches.0.block-1_att.bn.running_var', 
'input_branches.0.block-1_att.bn.num_batches_tracked', 

'input_branches.1.init_bn.weight', 
'input_branches.1.init_bn.bias', 
'input_branches.1.init_bn.running_mean', 
'input_branches.1.init_bn.running_var',
'input_branches.1.init_bn.num_batches_tracked', 

'input_branches.1.stem_scn.conv.A', 
'input_branches.1.stem_scn.conv.edge', 

'input_branches.1.stem_scn.conv.gcn.weight', 
'input_branches.1.stem_scn.conv.gcn.bias', 

'input_branches.1.stem_scn.bn.weight', 
'input_branches.1.stem_scn.bn.bias', 
'input_branches.1.stem_scn.bn.running_mean', 
'input_branches.1.stem_scn.bn.running_var',
'input_branches.1.stem_scn.bn.num_batches_tracked', 

'input_branches.1.stem_scn.residual.0.weight', 
'input_branches.1.stem_scn.residual.0.bias', 

'input_branches.1.stem_scn.residual.1.weight', 
'input_branches.1.stem_scn.residual.1.bias', 
'input_branches.1.stem_scn.residual.1.running_mean', 
'input_branches.1.stem_scn.residual.1.running_var', 
'input_branches.1.stem_scn.residual.1.num_batches_tracked', 

'input_branches.1.stem_tcn.conv.weight', 
'input_branches.1.stem_tcn.conv.bias', 

'input_branches.1.stem_tcn.bn.weight', 
'input_branches.1.stem_tcn.bn.bias', 
'input_branches.1.stem_tcn.bn.running_mean', 
'input_branches.1.stem_tcn.bn.running_var', 
'input_branches.1.stem_tcn.bn.num_batches_tracked', 

'input_branches.1.block-0_scn.conv.A', 
'input_branches.1.block-0_scn.conv.edge', 

'input_branches.1.block-0_scn.conv.gcn.weight',
'input_branches.1.block-0_scn.conv.gcn.bias', 

'input_branches.1.block-0_scn.bn.weight',
'input_branches.1.block-0_scn.bn.bias',
'input_branches.1.block-0_scn.bn.running_mean', 
'input_branches.1.block-0_scn.bn.running_var', 
'input_branches.1.block-0_scn.bn.num_batches_tracked', 

'input_branches.1.block-0_scn.residual.0.weight', 
'input_branches.1.block-0_scn.residual.0.bias', 

'input_branches.1.block-0_scn.residual.1.weight', 
'input_branches.1.block-0_scn.residual.1.bias', 
'input_branches.1.block-0_scn.residual.1.running_mean', 
'input_branches.1.block-0_scn.residual.1.running_var',
'input_branches.1.block-0_scn.residual.1.num_batches_tracked', 

'input_branches.1.block-0_att.att.fcn.0.weight', 
'input_branches.1.block-0_att.att.fcn.0.bias', 

'input_branches.1.block-0_att.att.fcn.1.weight', 
'input_branches.1.block-0_att.att.fcn.1.bias', 
'input_branches.1.block-0_att.att.fcn.1.running_mean', 
'input_branches.1.block-0_att.att.fcn.1.running_var', 
'input_branches.1.block-0_att.att.fcn.1.num_batches_tracked',

'input_branches.1.block-0_att.att.conv_t.weight',
'input_branches.1.block-0_att.att.conv_t.bias', 

'input_branches.1.block-0_att.att.conv_v.weight', 
'input_branches.1.block-0_att.att.conv_v.bias', 

'input_branches.1.block-0_att.bn.weight', 
'input_branches.1.block-0_att.bn.bias', 
'input_branches.1.block-0_att.bn.running_mean', 
'input_branches.1.block-0_att.bn.running_var', 
'input_branches.1.block-0_att.bn.num_batches_tracked', 

'input_branches.1.block-1_scn.conv.A', 
'input_branches.1.block-1_scn.conv.edge', 

'input_branches.1.block-1_scn.conv.gcn.weight', 
'input_branches.1.block-1_scn.conv.gcn.bias', 

'input_branches.1.block-1_scn.bn.weight',
'input_branches.1.block-1_scn.bn.bias',
'input_branches.1.block-1_scn.bn.running_mean', 
'input_branches.1.block-1_scn.bn.running_var', 
'input_branches.1.block-1_scn.bn.num_batches_tracked', 

'input_branches.1.block-1_scn.residual.0.weight', 
'input_branches.1.block-1_scn.residual.0.bias',

'input_branches.1.block-1_scn.residual.1.weight', 
'input_branches.1.block-1_scn.residual.1.bias', 
'input_branches.1.block-1_scn.residual.1.running_mean', 
'input_branches.1.block-1_scn.residual.1.running_var',
'input_branches.1.block-1_scn.residual.1.num_batches_tracked',

'input_branches.1.block-1_att.att.fcn.0.weight',
'input_branches.1.block-1_att.att.fcn.0.bias', 

'input_branches.1.block-1_att.att.fcn.1.weight',
'input_branches.1.block-1_att.att.fcn.1.bias', 
'input_branches.1.block-1_att.att.fcn.1.running_mean', 
'input_branches.1.block-1_att.att.fcn.1.running_var', 
'input_branches.1.block-1_att.att.fcn.1.num_batches_tracked', 

'input_branches.1.block-1_att.att.conv_t.weight', 
'input_branches.1.block-1_att.att.conv_t.bias', 

'input_branches.1.block-1_att.att.conv_v.weight', 
'input_branches.1.block-1_att.att.conv_v.bias',

'input_branches.1.block-1_att.bn.weight', 
'input_branches.1.block-1_att.bn.bias', 
'input_branches.1.block-1_att.bn.running_mean', 
'input_branches.1.block-1_att.bn.running_var',
'input_branches.1.block-1_att.bn.num_batches_tracked', 

'input_branches.2.init_bn.weight', 
'input_branches.2.init_bn.bias', 
'input_branches.2.init_bn.running_mean', 
'input_branches.2.init_bn.running_var', 
'input_branches.2.init_bn.num_batches_tracked', 

'input_branches.2.stem_scn.conv.A', 
'input_branches.2.stem_scn.conv.edge', 

'input_branches.2.stem_scn.conv.gcn.weight',
'input_branches.2.stem_scn.conv.gcn.bias',

'input_branches.2.stem_scn.bn.weight',
'input_branches.2.stem_scn.bn.bias',
'input_branches.2.stem_scn.bn.running_mean', 
'input_branches.2.stem_scn.bn.running_var', 
'input_branches.2.stem_scn.bn.num_batches_tracked', 

'input_branches.2.stem_scn.residual.0.weight', 
'input_branches.2.stem_scn.residual.0.bias', 

'input_branches.2.stem_scn.residual.1.weight', 
'input_branches.2.stem_scn.residual.1.bias', 
'input_branches.2.stem_scn.residual.1.running_mean', 
'input_branches.2.stem_scn.residual.1.running_var', 
'input_branches.2.stem_scn.residual.1.num_batches_tracked', 

'input_branches.2.stem_tcn.conv.weight', 
'input_branches.2.stem_tcn.conv.bias',

'input_branches.2.stem_tcn.bn.weight', 
'input_branches.2.stem_tcn.bn.bias', 
'input_branches.2.stem_tcn.bn.running_mean', 
'input_branches.2.stem_tcn.bn.running_var', 
'input_branches.2.stem_tcn.bn.num_batches_tracked', 

'input_branches.2.block-0_scn.conv.A', 
'input_branches.2.block-0_scn.conv.edge', 

'input_branches.2.block-0_scn.conv.gcn.weight', 
'input_branches.2.block-0_scn.conv.gcn.bias', 

'input_branches.2.block-0_scn.bn.weight', 
'input_branches.2.block-0_scn.bn.bias', 
'input_branches.2.block-0_scn.bn.running_mean', 
'input_branches.2.block-0_scn.bn.running_var', 
'input_branches.2.block-0_scn.bn.num_batches_tracked', 

'input_branches.2.block-0_scn.residual.0.weight', 
'input_branches.2.block-0_scn.residual.0.bias', 

'input_branches.2.block-0_scn.residual.1.weight', 
'input_branches.2.block-0_scn.residual.1.bias', 
'input_branches.2.block-0_scn.residual.1.running_mean', 
'input_branches.2.block-0_scn.residual.1.running_var', 
'input_branches.2.block-0_scn.residual.1.num_batches_tracked', 

'input_branches.2.block-0_att.att.fcn.0.weight',
'input_branches.2.block-0_att.att.fcn.0.bias', 

'input_branches.2.block-0_att.att.fcn.1.weight',
'input_branches.2.block-0_att.att.fcn.1.bias',
'input_branches.2.block-0_att.att.fcn.1.running_mean', 
'input_branches.2.block-0_att.att.fcn.1.running_var', 
'input_branches.2.block-0_att.att.fcn.1.num_batches_tracked', 

'input_branches.2.block-0_att.att.conv_t.weight', 
'input_branches.2.block-0_att.att.conv_t.bias', 

'input_branches.2.block-0_att.att.conv_v.weight', 
'input_branches.2.block-0_att.att.conv_v.bias', 

'input_branches.2.block-0_att.bn.weight',
'input_branches.2.block-0_att.bn.bias', 
'input_branches.2.block-0_att.bn.running_mean', 
'input_branches.2.block-0_att.bn.running_var', 
'input_branches.2.block-0_att.bn.num_batches_tracked', 

'input_branches.2.block-1_scn.conv.A', 
'input_branches.2.block-1_scn.conv.edge', 

'input_branches.2.block-1_scn.conv.gcn.weight', 
'input_branches.2.block-1_scn.conv.gcn.bias', 

'input_branches.2.block-1_scn.bn.weight',
'input_branches.2.block-1_scn.bn.bias', 
'input_branches.2.block-1_scn.bn.running_mean', 
'input_branches.2.block-1_scn.bn.running_var',
'input_branches.2.block-1_scn.bn.num_batches_tracked', 

'input_branches.2.block-1_scn.residual.0.weight', 
'input_branches.2.block-1_scn.residual.0.bias',

'input_branches.2.block-1_scn.residual.1.weight', 
'input_branches.2.block-1_scn.residual.1.bias',
'input_branches.2.block-1_scn.residual.1.running_mean', 
'input_branches.2.block-1_scn.residual.1.running_var', 
'input_branches.2.block-1_scn.residual.1.num_batches_tracked', 

'input_branches.2.block-1_att.att.fcn.0.weight', 
'input_branches.2.block-1_att.att.fcn.0.bias',

'input_branches.2.block-1_att.att.fcn.1.weight', 
'input_branches.2.block-1_att.att.fcn.1.bias', 
'input_branches.2.block-1_att.att.fcn.1.running_mean',
'input_branches.2.block-1_att.att.fcn.1.running_var',
'input_branches.2.block-1_att.att.fcn.1.num_batches_tracked', 

'input_branches.2.block-1_att.att.conv_t.weight', 
'input_branches.2.block-1_att.att.conv_t.bias', 

'input_branches.2.block-1_att.att.conv_v.weight', 
'input_branches.2.block-1_att.att.conv_v.bias', 

'input_branches.2.block-1_att.bn.weight', 
'input_branches.2.block-1_att.bn.bias', 
'input_branches.2.block-1_att.bn.running_mean',
'input_branches.2.block-1_att.bn.running_var', 
'input_branches.2.block-1_att.bn.num_batches_tracked',

'main_stream.block-0_scn.conv.A', 
'main_stream.block-0_scn.conv.edge', 

'main_stream.block-0_scn.conv.gcn.weight', 
'main_stream.block-0_scn.conv.gcn.bias',

'main_stream.block-0_scn.bn.weight',
'main_stream.block-0_scn.bn.bias',
'main_stream.block-0_scn.bn.running_mean', 
'main_stream.block-0_scn.bn.running_var', 
'main_stream.block-0_scn.bn.num_batches_tracked', 

'main_stream.block-0_scn.residual.0.weight', 
'main_stream.block-0_scn.residual.0.bias', 

'main_stream.block-0_scn.residual.1.weight', 
'main_stream.block-0_scn.residual.1.bias', 
'main_stream.block-0_scn.residual.1.running_mean', 
'main_stream.block-0_scn.residual.1.running_var', 
'main_stream.block-0_scn.residual.1.num_batches_tracked', 

'main_stream.block-0_tcn-0.depth_conv1.0.weight',
'main_stream.block-0_tcn-0.depth_conv1.0.bias',

'main_stream.block-0_tcn-0.depth_conv1.1.weight', 
'main_stream.block-0_tcn-0.depth_conv1.1.bias', 
'main_stream.block-0_tcn-0.depth_conv1.1.running_mean', 
'main_stream.block-0_tcn-0.depth_conv1.1.running_var', 
'main_stream.block-0_tcn-0.depth_conv1.1.num_batches_tracked',

'main_stream.block-0_tcn-0.point_conv1.0.weight',
'main_stream.block-0_tcn-0.point_conv1.0.bias',

'main_stream.block-0_tcn-0.point_conv1.1.weight', 
'main_stream.block-0_tcn-0.point_conv1.1.bias',
'main_stream.block-0_tcn-0.point_conv1.1.running_mean', 
'main_stream.block-0_tcn-0.point_conv1.1.running_var', 
'main_stream.block-0_tcn-0.point_conv1.1.num_batches_tracked',

'main_stream.block-0_tcn-0.point_conv2.0.weight', 
'main_stream.block-0_tcn-0.point_conv2.0.bias', 

'main_stream.block-0_tcn-0.point_conv2.1.weight', 
'main_stream.block-0_tcn-0.point_conv2.1.bias', 
'main_stream.block-0_tcn-0.point_conv2.1.running_mean', 
'main_stream.block-0_tcn-0.point_conv2.1.running_var', 
'main_stream.block-0_tcn-0.point_conv2.1.num_batches_tracked',

'main_stream.block-0_tcn-0.depth_conv2.0.weight', 
'main_stream.block-0_tcn-0.depth_conv2.0.bias', 

'main_stream.block-0_tcn-0.depth_conv2.1.weight', 
'main_stream.block-0_tcn-0.depth_conv2.1.bias', 
'main_stream.block-0_tcn-0.depth_conv2.1.running_mean', 
'main_stream.block-0_tcn-0.depth_conv2.1.running_var', 
'main_stream.block-0_tcn-0.depth_conv2.1.num_batches_tracked', 

'main_stream.block-0_tcn-0.residual.0.weight',
'main_stream.block-0_tcn-0.residual.0.bias',

'main_stream.block-0_tcn-0.residual.1.weight', 
'main_stream.block-0_tcn-0.residual.1.bias', 
'main_stream.block-0_tcn-0.residual.1.running_mean', 
'main_stream.block-0_tcn-0.residual.1.running_var',
'main_stream.block-0_tcn-0.residual.1.num_batches_tracked', 

'main_stream.block-0_att.att.fcn.0.weight', 
'main_stream.block-0_att.att.fcn.0.bias', 

'main_stream.block-0_att.att.fcn.1.weight', 
'main_stream.block-0_att.att.fcn.1.bias', 
'main_stream.block-0_att.att.fcn.1.running_mean', 
'main_stream.block-0_att.att.fcn.1.running_var', 
'main_stream.block-0_att.att.fcn.1.num_batches_tracked', 

'main_stream.block-0_att.att.conv_t.weight', 
'main_stream.block-0_att.att.conv_t.bias', 

'main_stream.block-0_att.att.conv_v.weight',
'main_stream.block-0_att.att.conv_v.bias', 

'main_stream.block-0_att.bn.weight', 
'main_stream.block-0_att.bn.bias', 
'main_stream.block-0_att.bn.running_mean',
'main_stream.block-0_att.bn.running_var', 
'main_stream.block-0_att.bn.num_batches_tracked', 

'main_stream.block-1_scn.conv.A',
'main_stream.block-1_scn.conv.edge', 

'main_stream.block-1_scn.conv.gcn.weight', 
'main_stream.block-1_scn.conv.gcn.bias', 

'main_stream.block-1_scn.bn.weight',
'main_stream.block-1_scn.bn.bias',
'main_stream.block-1_scn.bn.running_mean', 
'main_stream.block-1_scn.bn.running_var',
'main_stream.block-1_scn.bn.num_batches_tracked', 

'main_stream.block-1_scn.residual.0.weight',
'main_stream.block-1_scn.residual.0.bias', 

'main_stream.block-1_scn.residual.1.weight', 
'main_stream.block-1_scn.residual.1.bias', 
'main_stream.block-1_scn.residual.1.running_mean', 
'main_stream.block-1_scn.residual.1.running_var', 
'main_stream.block-1_scn.residual.1.num_batches_tracked', 

'main_stream.block-1_tcn-0.depth_conv1.0.weight',
'main_stream.block-1_tcn-0.depth_conv1.0.bias', 

'main_stream.block-1_tcn-0.depth_conv1.1.weight',
'main_stream.block-1_tcn-0.depth_conv1.1.bias', 
'main_stream.block-1_tcn-0.depth_conv1.1.running_mean',
'main_stream.block-1_tcn-0.depth_conv1.1.running_var',
'main_stream.block-1_tcn-0.depth_conv1.1.num_batches_tracked', 

'main_stream.block-1_tcn-0.point_conv1.0.weight', 
'main_stream.block-1_tcn-0.point_conv1.0.bias', 

'main_stream.block-1_tcn-0.point_conv1.1.weight', 
'main_stream.block-1_tcn-0.point_conv1.1.bias', 
'main_stream.block-1_tcn-0.point_conv1.1.running_mean', 
'main_stream.block-1_tcn-0.point_conv1.1.running_var',
'main_stream.block-1_tcn-0.point_conv1.1.num_batches_tracked', 

'main_stream.block-1_tcn-0.point_conv2.0.weight',
'main_stream.block-1_tcn-0.point_conv2.0.bias', 

'main_stream.block-1_tcn-0.point_conv2.1.weight',
'main_stream.block-1_tcn-0.point_conv2.1.bias',
'main_stream.block-1_tcn-0.point_conv2.1.running_mean',
'main_stream.block-1_tcn-0.point_conv2.1.running_var',
'main_stream.block-1_tcn-0.point_conv2.1.num_batches_tracked', 

'main_stream.block-1_tcn-0.depth_conv2.0.weight',
'main_stream.block-1_tcn-0.depth_conv2.0.bias',

'main_stream.block-1_tcn-0.depth_conv2.1.weight',
'main_stream.block-1_tcn-0.depth_conv2.1.bias',
'main_stream.block-1_tcn-0.depth_conv2.1.running_mean',
'main_stream.block-1_tcn-0.depth_conv2.1.running_var',
'main_stream.block-1_tcn-0.depth_conv2.1.num_batches_tracked',

'main_stream.block-1_tcn-0.residual.0.weight',
'main_stream.block-1_tcn-0.residual.0.bias',

'main_stream.block-1_tcn-0.residual.1.weight',
'main_stream.block-1_tcn-0.residual.1.bias', 
'main_stream.block-1_tcn-0.residual.1.running_mean',
'main_stream.block-1_tcn-0.residual.1.running_var',
'main_stream.block-1_tcn-0.residual.1.num_batches_tracked', 

'main_stream.block-1_att.att.fcn.0.weight', 
'main_stream.block-1_att.att.fcn.0.bias',

'main_stream.block-1_att.att.fcn.1.weight',
'main_stream.block-1_att.att.fcn.1.bias',
'main_stream.block-1_att.att.fcn.1.running_mean',
'main_stream.block-1_att.att.fcn.1.running_var',
'main_stream.block-1_att.att.fcn.1.num_batches_tracked', 

'main_stream.block-1_att.att.conv_t.weight',
'main_stream.block-1_att.att.conv_t.bias', 

'main_stream.block-1_att.att.conv_v.weight', 
'main_stream.block-1_att.att.conv_v.bias',

'main_stream.block-1_att.bn.weight', 
'main_stream.block-1_att.bn.bias', 
'main_stream.block-1_att.bn.running_mean', 
'main_stream.block-1_att.bn.running_var', 
'main_stream.block-1_att.bn.num_batches_tracked', 
'classifier.fc.weight',
'classifier.fc.bias'
    ]
    fc_names_1 = []

    for key in torch_dict:

        if (type(torch_dict[key]) == int):
            print('yes for int')
            print(torch_dict[key])
            weight = torch_dict[key]
        elif (type(torch_dict[key]) == bool):
            print('yes for bool')
            print(torch_dict[key])
            weight = torch_dict[key]
        else:
            weight = torch_dict[key].cpu().detach().numpy()
            flag = [i in key for i in fc_names] 
            if any(flag):
                print("weight {} need to be trans".format(key))
                if (key[-7:] == ".weight"):
                    pass
                elif (key[-5:] == ".bias"):
                    pass
                elif (key[-13:] == ".running_mean"):
                    key = key[:-12] + "_mean"
                elif (key[-12:] == ".running_var"):
                    key = key[:-11] + "_variance"
                elif (key[-20:] == ".num_batches_tracked"):
                    pass
                elif (key[-2:] == ".A"):
                    pass
                elif (key[-5:] == ".edge"):
                    pass
                else: 
                    print("error")

            flag_1 = [i in key for i in fc_names_1]
            if any(flag_1):
                print("weight {} need to be trans".format(key))
                if (key[-7:] == ".weight"):
                    weight = weight.transpose()
                else:
                    print('error1')
                    print("-----"+key)
        paddle_dict[key] = weight

    print("before transfer......")
    paddle_all_dict = {}
    paddle_all_dict['model'] = paddle_dict
    paddle_all_dict['best_state'] = torch_all_dict['best_state']
    
    paddle.save(paddle_all_dict, output_fp)
    print('success')

transfer()