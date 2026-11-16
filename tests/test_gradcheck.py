"""Every op's analytic gradient vs. central finite differences (float64)."""

import numpy as np
import pytest
from op_specs import SPEC_IDS, SPECS

from tensorgrad import Tensor
from tensorgrad.testing import gradcheck


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_gradcheck(spec):
    rng = np.random.default_rng(0)
    inputs = [Tensor(a, requires_grad=True) for a in spec.sample(rng)]
    gradcheck(spec.tg, inputs, eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_float32_is_preserved(spec):
    """No op silently promotes float32 activations to float64."""
    rng = np.random.default_rng(0)
    inputs = [Tensor(a.astype(np.float32), requires_grad=True) for a in spec.sample(rng)]
    out = spec.tg(*inputs)
    assert out.dtype == np.float32, f"{spec.name} returned {out.dtype}"
    out.sum().backward()
    for x in inputs:
        assert x.grad is not None and x.grad.dtype == np.float32
