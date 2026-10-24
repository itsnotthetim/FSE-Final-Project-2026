import pytest
import torch

from fse_ops import soft_shrink


def spec(x):
    return torch.sign(x) * (x.abs() - 1).clamp(min=0)


# kind 1: shape
def test_output_shape():
    x = torch.randn(4, 3, 2)
    assert soft_shrink(x).shape == x.shape


# kind 2: dtype
def test_output_dtype():
    assert soft_shrink(torch.randn(5)).dtype == torch.float32


# kind 3: exact numeric input vs the spec line
def test_matches_spec():
    x = torch.tensor([-3.0, 0.5, 2.5])
    assert torch.equal(soft_shrink(x), spec(x))


# kind 3 (second): leading batch dimension of 4, like the grader's A3
def test_batch_dim():
    x = torch.tensor([-3.0, 0.5, 2.5]).repeat(4, 1)
    assert torch.equal(soft_shrink(x), spec(x))


# kind 4: a fake op cannot survive this
def test_values_differ_per_element():
    x = torch.tensor([-1.5, 4.0, -0.25])
    out = soft_shrink(x)
    assert torch.equal(out, torch.tensor([-0.5, 3.0, 0.0]))
    assert not torch.equal(out, x)
    assert len(set(out.tolist())) == 3


# kind 5: error path (TORCH_CHECK -> RuntimeError)
def test_rejects_int_tensor():
    with pytest.raises(RuntimeError):
        soft_shrink(torch.tensor([1, 2, 3]))