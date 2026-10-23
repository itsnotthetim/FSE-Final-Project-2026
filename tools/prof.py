import torch

from fse_ops import soft_shrink

soft_shrink(torch.randn(10_000_000))