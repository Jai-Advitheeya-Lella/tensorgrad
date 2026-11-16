"""Autodiff engine semantics: accumulation, graph shapes, grad modes, broadcasting."""

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import Tensor
from tensorgrad.tensor import unbroadcast


def test_scalar_chain_rule_by_hand():
    # f = (a*b + c)^2 * tanh(a)
    a = Tensor(1.5, requires_grad=True, dtype=np.float64)
    b = Tensor(-0.7, requires_grad=True, dtype=np.float64)
    c = Tensor(0.3, requires_grad=True, dtype=np.float64)
    f = (a * b + c) ** 2 * a.tanh()
    f.backward()
    u, t = 1.5 * -0.7 + 0.3, np.tanh(1.5)
    assert np.isclose(a.grad, 2 * u * -0.7 * t + u**2 * (1 - t**2))
    assert np.isclose(b.grad, 2 * u * 1.5 * t)
    assert np.isclose(c.grad, 2 * u * t)


def test_diamond_graph_accumulates():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    y = x * x      # used twice below
    z = y + y * x  # dz/dx = 2x + 3x^2
    z.sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 3 * x.data**2)


def test_grads_accumulate_across_backward_calls():
    x = Tensor(np.ones(3), requires_grad=True)
    (x * 2).sum().backward()
    (x * 3).sum().backward()
    np.testing.assert_allclose(x.grad, 5 * np.ones(3))


def test_no_grad_builds_no_graph():
    x = Tensor(np.ones(3), requires_grad=True)
    with tg.no_grad():
        y = x * 2
    assert not y.requires_grad and y._ctx is None

    @tg.no_grad()
    def f(t):
        return t + 1

    assert f(x)._ctx is None
    assert tg.is_grad_enabled()


def test_detach_blocks_gradient():
    x = Tensor(np.ones(3), requires_grad=True)
    y = (x * 2).detach() * x
    y.sum().backward()
    np.testing.assert_allclose(x.grad, 2 * np.ones(3))


def test_retain_grad_on_intermediate():
    x = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    h = (x * 3).retain_grad()
    (h**2).sum().backward()
    np.testing.assert_allclose(h.grad, 2 * h.data)
    np.testing.assert_allclose(x.grad, 6 * h.data)


def test_constants_get_no_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    c = Tensor(np.full(3, 4.0))
    (x * c).sum().backward()
    assert c.grad is None


def test_non_scalar_backward_requires_seed():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()
    (x * 2).backward(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(x.grad, [2.0, 4.0, 6.0])


def test_deep_graph_has_no_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True, dtype=np.float64)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.backward()
    assert np.isclose(x.grad, 1.0001**5000)


@pytest.mark.parametrize(
    "grad_shape,target",
    [((2, 3, 4), (4,)), ((2, 3, 4), (3, 1)), ((2, 3, 4), (1, 1, 1)), ((5,), ()), ((2, 3), (2, 3))],
)
def test_unbroadcast_matches_broadcast_adjoint(grad_shape, target):
    rng = np.random.default_rng(0)
    g = rng.standard_normal(grad_shape)
    v = rng.standard_normal(target)
    # <g, broadcast(v)> == <unbroadcast(g), v>  (adjointness)
    lhs = (g * np.broadcast_to(v, grad_shape)).sum()
    red = unbroadcast(g, target)
    assert red.shape == tuple(target)
    assert np.isclose(lhs, (red * v).sum())


def test_int_tensors_cannot_require_grad():
    with pytest.raises(TypeError):
        Tensor(np.arange(3), requires_grad=True)


def test_python_floats_use_default_dtype():
    assert Tensor([1.0, 2.0]).dtype == np.float32
    assert Tensor(np.zeros(2)).dtype == np.float64  # explicit ndarray keeps its dtype


def test_ndarray_on_left_dispatches_to_tensor():
    x = Tensor(np.ones(2), requires_grad=True)
    y = np.array([2.0, 3.0]) * x
    assert isinstance(y, Tensor)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 3.0])
