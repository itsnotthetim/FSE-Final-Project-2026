#!/bin/bash
rm -rf /tmp/smoke-build /tmp/smoke-run dist &&
python3 -m venv /tmp/smoke-build &&
/tmp/smoke-build/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu &&
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu /tmp/smoke-build/bin/pip wheel . --no-deps -w dist &&
ls dist | grep linux &&
python3 -m venv /tmp/smoke-run &&
/tmp/smoke-run/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu &&
/tmp/smoke-run/bin/pip install dist/*.whl &&
cd /tmp && /tmp/smoke-run/bin/python -c 'import torch; from fse_ops import soft_shrink; print(soft_shrink(torch.tensor([-3.0, 0.5, 2.5])))' &&
echo OK
