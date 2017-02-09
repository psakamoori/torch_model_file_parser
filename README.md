
# TorchModelFileParser

# Overview
Parser .t7 torch model file to extract Layer information, Kernel, Stride and Padding.
This helps in analyzing work loads on VPU's 

# Pre-requisites:						       
 Machine with lua, torch and supporting Luarocks isntalled           
 Ref: www.torch.ch for more information  
 
# Script
Python t7_parser.py

# Sample Output of nn4.small2.v1.t7

nn.SpatialConvolutionMM(3 -> 64, 7x7, 2,2, 3,3)

nn.SpatialBatchNormalization

nn.ReLU

nn.SpatialMaxPooling(3x3, 2,2, 1,1)

nn.SpatialCrossMapLRN

nn.SpatialConvolutionMM(64 -> 64, 1x1)

nn.SpatialBatchNormalization

nn.ReLU

nn.SpatialConvolutionMM(64 -> 192, 3x3, 1,1, 1,1)

nn.SpatialBatchNormalization

nn.ReLU

nn.SpatialCrossMapLRN

nn.SpatialMaxPooling(3x3, 2,2, 1,1)

.........
  
