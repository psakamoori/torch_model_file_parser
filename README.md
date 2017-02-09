
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
nn.Inception @ nn.DepthConcat {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(192 -> 96, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(96 -> 128, 3x3, 1,1, 1,1)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (2): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(192 -> 16, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(16 -> 32, 5x5, 1,1, 2,2)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (3): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> output]
    |      (1): nn.SpatialMaxPooling(3x3, 2,2)
    |      (2): nn.SpatialConvolution(192 -> 32, 1x1)
    |      (3): nn.SpatialBatchNormalization
    |      (4): nn.ReLU
    |    }
     `-> (4): nn.Sequential {
           [input -> (1) -> (2) -> (3) -> output]
           (1): nn.SpatialConvolution(192 -> 64, 1x1)
           (2): nn.SpatialBatchNormalization
           (3): nn.ReLU
         }
     ... -> output
}
nn.Inception @ nn.DepthConcat {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(256 -> 96, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(96 -> 128, 3x3, 1,1, 1,1)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (2): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(256 -> 32, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(32 -> 64, 5x5, 1,1, 2,2)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (3): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> output]
    |      (1): nn.Sequential {
    |        [input -> (1) -> (2) -> (3) -> (4) -> output]
    |        (1): nn.Square
    |        (2): nn.SpatialAveragePooling(3x3, 3,3)
    |        (3): nn.MulConstant
    |        (4): nn.Sqrt
    |      }
    |      (2): nn.SpatialConvolution(256 -> 64, 1x1)
    |      (3): nn.SpatialBatchNormalization
    |      (4): nn.ReLU
    |    }
     `-> (4): nn.Sequential {
           [input -> (1) -> (2) -> (3) -> output]
           (1): nn.SpatialConvolution(256 -> 64, 1x1)
           (2): nn.SpatialBatchNormalization
           (3): nn.ReLU
         }
     ... -> output
}
nn.Inception @ nn.DepthConcat {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(320 -> 128, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(128 -> 256, 3x3, 2,2, 1,1)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (2): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(320 -> 32, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(32 -> 64, 5x5, 2,2, 2,2)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
     `-> (3): nn.Sequential {
           [input -> (1) -> output]
           (1): nn.SpatialMaxPooling(3x3, 2,2)
         }
     ... -> output
}
nn.Inception @ nn.DepthConcat {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(640 -> 96, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(96 -> 192, 3x3, 1,1, 1,1)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (2): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(640 -> 32, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(32 -> 64, 5x5, 1,1, 2,2)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (3): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> output]
    |      (1): nn.Sequential {
    |        [input -> (1) -> (2) -> (3) -> (4) -> output]
    |        (1): nn.Square
    |        (2): nn.SpatialAveragePooling(3x3, 3,3)
    |        (3): nn.MulConstant
    |        (4): nn.Sqrt
    |      }
    |      (2): nn.SpatialConvolution(640 -> 128, 1x1)
    |      (3): nn.SpatialBatchNormalization
    |      (4): nn.ReLU
    |    }
     `-> (4): nn.Sequential {
           [input -> (1) -> (2) -> (3) -> output]
           (1): nn.SpatialConvolution(640 -> 256, 1x1)
           (2): nn.SpatialBatchNormalization
           (3): nn.ReLU
         }
     ... -> output
}
nn.Inception @ nn.DepthConcat {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(640 -> 160, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(160 -> 256, 3x3, 2,2, 1,1)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (2): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(640 -> 64, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(64 -> 128, 5x5, 2,2, 2,2)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
     `-> (3): nn.Sequential {
           [input -> (1) -> output]
           (1): nn.SpatialMaxPooling(3x3, 2,2)
         }
     ... -> output
}
nn.Inception @ nn.DepthConcat {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(1024 -> 96, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(96 -> 384, 3x3, 1,1, 1,1)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (2): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> output]
    |      (1): nn.Sequential {
    |        [input -> (1) -> (2) -> (3) -> (4) -> output]
    |        (1): nn.Square
    |        (2): nn.SpatialAveragePooling(3x3, 3,3)
    |        (3): nn.MulConstant
    |        (4): nn.Sqrt
    |      }
    |      (2): nn.SpatialConvolution(1024 -> 96, 1x1)
    |      (3): nn.SpatialBatchNormalization
    |      (4): nn.ReLU
    |    }
     `-> (3): nn.Sequential {
           [input -> (1) -> (2) -> (3) -> output]
           (1): nn.SpatialConvolution(1024 -> 256, 1x1)
           (2): nn.SpatialBatchNormalization
           (3): nn.ReLU
         }
     ... -> output
}
nn.Reshape(736x3x3)
nn.Inception @ nn.DepthConcat {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
    |      (1): nn.SpatialConvolution(736 -> 96, 1x1)
    |      (2): nn.SpatialBatchNormalization
    |      (3): nn.ReLU
    |      (4): nn.SpatialConvolution(96 -> 384, 3x3, 1,1, 1,1)
    |      (5): nn.SpatialBatchNormalization
    |      (6): nn.ReLU
    |    }
    |`-> (2): nn.Sequential {
    |      [input -> (1) -> (2) -> (3) -> (4) -> output]
    |      (1): nn.SpatialMaxPooling(3x3, 2,2)
    |      (2): nn.SpatialConvolution(736 -> 96, 1x1)
    |      (3): nn.SpatialBatchNormalization
    |      (4): nn.ReLU
    |    }
     `-> (3): nn.Sequential {
           [input -> (1) -> (2) -> (3) -> output]
           (1): nn.SpatialConvolution(736 -> 256, 1x1)
           (2): nn.SpatialBatchNormalization
           (3): nn.ReLU
         }
     ... -> output
}
nn.SpatialAveragePooling(3x3, 1,1)
nn.Reshape(736)
nn.View(736)
nn.Linear(736 -> 128)
nn.Normalize(2)

