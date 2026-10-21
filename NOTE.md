# Project note

## Bug we hit
<what happened, how it surfaced (red test / traceback / pdb session),
the fix, and the commit hash from `git log --oneline`>

## Case the op does not handle
`soft_shrink` accepts float32 only. float64 and float16 tensors are
rejected by `TORCH_CHECK` and raise `RuntimeError` instead of being converted.

## Measurement
Command: `python3 -m cProfile -s cumtime -c "import torch; from fse_ops import soft_shrink; soft_shrink(torch.randn(10_000_000))"`
Result: <the number it printed>