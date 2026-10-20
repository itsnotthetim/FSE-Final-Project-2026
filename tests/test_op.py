import pytest
import torch
from fse_ops import soft_shrink


def spec(x):
    return torch.sign(x) * (x.abs() - 1).clamp(min=0)


def test_shape():
    x = torch.randn(4, 3, 2)
    assert soft_shrink(x).shape == x.shape


def test_dtype():
    assert soft_shrink(torch.randn(5)).dtype == torch.float32


def test_exact_numeric():
    x = torch.tensor([-3.0, 0.5, 2.5])
    assert torch.equal(soft_shrink(x), spec(x))


def test_not_a_fake_op():
    x = torch.tensor([-3.0, 0.5, 2.5])
    out = soft_shrink(x)
    assert torch.equal(out, torch.tensor([-2.0, 0.0, 1.5]))
    assert not torch.equal(out, x)
    assert len(torch.unique(out)) == 3


def test_int_rejected():
    with pytest.raises(RuntimeError):
        soft_shrink(torch.tensor([1, 2, 3]))