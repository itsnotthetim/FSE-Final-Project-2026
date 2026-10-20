import torch  # noqa: F401  (loads libtorch before the extension)

from ._C import soft_shrink

__all__ = ["soft_shrink"]
