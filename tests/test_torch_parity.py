"""Every op's forward output and input gradients vs. PyTorch (float64, CPU)."""

import numpy as np
import pytest
from op_specs import SPEC_IDS, SPECS

from tensorgrad import Tensor

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_matches_torch(spec):
    rng = np.random.default_rng(42)
    arrays = spec.sample(rng)
    tg_in = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    th_in = [torch.tensor(a.copy(), dtype=torch.float64, requires_grad=True) for a in arrays]

    tg_out = spec.tg(*tg_in)
    th_out = spec.th(*th_in)
    np.testing.assert_allclose(tg_out.data, th_out.detach().numpy(), atol=1e-10, rtol=1e-8,
                               err_msg=f"{spec.name}: forward mismatch")

    upstream = rng.standard_normal(tg_out.shape)
    tg_out.backward(upstream)
    th_out.backward(torch.from_numpy(upstream))
    for i, (a, b) in enumerate(zip(tg_in, th_in)):
        np.testing.assert_allclose(a.grad, b.grad.numpy(), atol=1e-10, rtol=1e-8,
                                   err_msg=f"{spec.name}: grad mismatch for input {i}")
