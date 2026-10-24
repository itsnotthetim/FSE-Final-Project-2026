# Project note

## Bug we hit
The cProfile command we first wrote in this note,
`python3 -m cProfile -s cumtime -c "..."`, failed with `no such option: -c`
on Python 3.10 and 3.12 (the `-c` flag exists only from Python 3.13).
A teammate hit it by running the note's command as written. The fix was to move
the code into `tools/prof.py` and run the script instead. Fixed in commit 56a658a.

## Case the op does not handle
`soft_shrink` accepts float32 only. float64 and float16 tensors are
rejected by `TORCH_CHECK` and raise `RuntimeError` instead of being converted.

## Measurement
Command: `python3 -m cProfile -s cumtime tools/prof.py`
Result: `soft_shrink` on 10,000,000 float32 values took 0.050 s
(`1 0.050 0.050 0.050 0.050 {built-in method fse_ops._C.soft_shrink}`).
The whole run took about 1.2 s (1.166 s in one run, 1.04 to 1.28 s over 3 runs,
Python 3.10, wheel in a fresh venv), mostly `import torch`.