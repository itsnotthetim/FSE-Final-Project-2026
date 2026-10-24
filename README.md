# fse_ops — soft shrink as a PyTorch C++ extension

FSE-Final-Project-2026. A custom PyTorch CPU operation written in C++
(`CppExtension`), shipped as a wheel.

## The function

```python
from fse_ops import soft_shrink
```

Python name: **`fse_ops.soft_shrink`**

**Formula in words:** for every element `x` of the input tensor,

- if `x > 1`, the result is `x - 1`;
- if `x < -1`, the result is `x + 1`;
- if `-1 <= x <= 1`, the result is `0`.

In other words, every value moves 1 toward zero and stops at zero. The output
has the same shape and dtype as the input, and each element depends only on the
element at the same position, so any shape works, including a leading batch
dimension.

**Specification line (pure PyTorch):**

```python
torch.sign(x) * (x.abs() - 1).clamp(min=0)
```

`soft_shrink(x)` returns exactly the same values as this line (`torch.equal`).

**Example:**

```python
>>> soft_shrink(torch.tensor([-3.0, 0.5, 2.5]))
tensor([-2.0000,  0.0000,  1.5000])
```

### Input rules

- The tensor must be **`torch.float32`**.
- The tensor must be on the **CPU**.

Anything else is rejected by `TORCH_CHECK` in the C++ code and raises
`RuntimeError`, for example an integer tensor:

```python
>>> soft_shrink(torch.tensor([1, 2, 3]))
RuntimeError: soft_shrink expects a float32 tensor, got Long
```

## Build, install, test

Run these from the repository directory, inside the course container
(`docker compose exec lab bash`). torch always comes from the CPU index.

**Build the wheel** into `dist/`:

```bash
python3 -m venv /tmp/build-venv
/tmp/build-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu /tmp/build-venv/bin/pip wheel . --no-deps -w dist
```

`pip wheel` builds in its own isolated environment and installs torch there
from `pyproject.toml`. `PIP_EXTRA_INDEX_URL` points it at the CPU index;
without it pip pulls the multi-gigabyte CUDA torch from PyPI and the build
takes over half an hour.

**Install** it into a fresh venv:

```bash
python3 -m venv /tmp/try-venv
/tmp/try-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/try-venv/bin/pip install dist/*.whl
```

**Try it** from a neutral directory:

```bash
cd /tmp && /tmp/try-venv/bin/python -c 'import torch; from fse_ops import soft_shrink; print(soft_shrink(torch.tensor([-3.0, 0.5, 2.5])))'
cd -
```

**Test** against the installed wheel (the tests are copied out of the source
tree, so they cannot import the sources instead of the wheel):

```bash
/tmp/try-venv/bin/pip install pytest
rm -rf /tmp/proj-tests && cp -r tests /tmp/proj-tests
cd /tmp && /tmp/try-venv/bin/python -m pytest -q /tmp/proj-tests
```

## make wheel / install / test

The `Makefile` runs the same commands. The venvs are created on first use
(torch is downloaded once) and reused afterwards.

```bash
make wheel    # build the wheel into dist/ (creates /tmp/build-venv)
make install  # build, then install dist/*.whl into /tmp/try-venv
make test     # build, install, copy tests to /tmp/proj-tests, run pytest from /tmp
make clean    # remove dist/, build/, *.egg-info and every __pycache__/
```

## Authors

See [AUTHORS](AUTHORS).
