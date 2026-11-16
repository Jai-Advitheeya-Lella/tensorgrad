"""Layers vs. PyTorch with copied weights, module bookkeeping, serialization."""

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import Tensor, nn
from tensorgrad.models import CharTransformer, ConvNet

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def float64():
    tg.set_default_dtype(np.float64)
    tg.manual_seed(0)
    torch.manual_seed(0)
    yield
    tg.set_default_dtype(np.float32)


def copy_params(tg_mod, th_mod, mapping=None):
    th_params = dict(th_mod.named_parameters())
    for name, p in tg_mod.named_parameters():
        th_name = mapping.get(name, name) if mapping else name
        with torch.no_grad():
            th_params[th_name].copy_(torch.from_numpy(p.data))


def compare_grads(tg_mod, th_mod, mapping=None, atol=1e-9):
    th_params = dict(th_mod.named_parameters())
    for name, p in tg_mod.named_parameters():
        th_name = mapping.get(name, name) if mapping else name
        np.testing.assert_allclose(p.grad, th_params[th_name].grad.numpy(), atol=atol, rtol=1e-7,
                                   err_msg=name)


def run_both(tg_mod, th_mod, x, mapping=None, atol=1e-9):
    copy_params(tg_mod, th_mod, mapping)
    xt = Tensor(x.copy(), requires_grad=True)
    xth = torch.tensor(x.copy(), requires_grad=True)
    out_tg = tg_mod(xt)
    out_th = th_mod(xth)
    np.testing.assert_allclose(out_tg.data, out_th.detach().numpy(), atol=atol, rtol=1e-7)
    up = np.random.default_rng(1).standard_normal(out_tg.shape)
    out_tg.backward(up)
    out_th.backward(torch.from_numpy(up))
    np.testing.assert_allclose(xt.grad, xth.grad.numpy(), atol=atol, rtol=1e-7)
    compare_grads(tg_mod, th_mod, mapping, atol)


def test_linear_matches_torch():
    run_both(nn.Linear(5, 3), torch.nn.Linear(5, 3).double(), np.random.randn(4, 5))


def test_conv_stack_matches_torch():
    tg_m = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(4, 3, 3, stride=2), nn.Flatten(), nn.Linear(3, 2))
    th_m = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                               torch.nn.Conv2d(4, 3, 3, stride=2), torch.nn.Flatten(),
                               torch.nn.Linear(3, 2)).double()
    run_both(tg_m, th_m, np.random.randn(2, 2, 8, 8))


def test_layernorm_matches_torch():
    run_both(nn.LayerNorm(6), torch.nn.LayerNorm(6).double(), np.random.randn(3, 4, 6))


def test_multihead_attention_matches_torch():
    d, h = 8, 2
    tg_m = nn.MultiHeadAttention(d, h, causal=True)
    th_m = torch.nn.MultiheadAttention(d, h, batch_first=True, bias=True).double()
    mapping = {"qkv.weight": "in_proj_weight", "qkv.bias": "in_proj_bias",
               "proj.weight": "out_proj.weight", "proj.bias": "out_proj.bias"}
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x, x, x, attn_mask=mask, need_weights=False)[0]

    wrapped = Wrapped(th_m)
    run_both(tg_m, wrapped, np.random.randn(2, 5, d), {k: "m." + v for k, v in mapping.items()})


def test_lstm_matches_torch():
    tg_m = nn.LSTM(3, 4)
    th_m = torch.nn.LSTM(3, 4, batch_first=True).double()
    mapping = {"weight_ih": "weight_ih_l0", "weight_hh": "weight_hh_l0",
               "bias_ih": "bias_ih_l0", "bias_hh": "bias_hh_l0"}

    class First(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    class FirstTG(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    run_both(FirstTG(tg_m), First(th_m), np.random.randn(2, 6, 3), {"m." + k: "m." + v for k, v in mapping.items()})


def test_batchnorm_running_stats_and_eval_match_torch():
    tg_bn, th_bn = nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3).double()
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal((4, 3, 5, 5)) * 2 + 1
        out_tg = tg_bn(Tensor(x))
        out_th = th_bn(torch.from_numpy(x))
        np.testing.assert_allclose(out_tg.data, out_th.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(tg_bn.running_mean, th_bn.running_mean.numpy(), atol=1e-12)
    np.testing.assert_allclose(tg_bn.running_var, th_bn.running_var.numpy(), atol=1e-12)
    tg_bn.eval(), th_bn.eval()
    x = rng.standard_normal((2, 3, 5, 5))
    np.testing.assert_allclose(tg_bn(Tensor(x)).data, th_bn(torch.from_numpy(x)).detach().numpy(), atol=1e-10)


def test_clip_grad_norm_matches_torch():
    rng = np.random.default_rng(0)
    gs = [rng.standard_normal(s) for s in [(3, 4), (5,)]]
    tg_ps = [Tensor(np.zeros_like(g), requires_grad=True) for g in gs]
    th_ps = [torch.zeros(g.shape, dtype=torch.float64, requires_grad=True) for g in gs]
    for p, q, g in zip(tg_ps, th_ps, gs):
        p.grad, q.grad = g.copy(), torch.from_numpy(g.copy())
    n1 = nn.clip_grad_norm_(tg_ps, 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(th_ps, 1.0)
    assert np.isclose(n1, n2.item())
    for p, q in zip(tg_ps, th_ps):
        np.testing.assert_allclose(p.grad, q.grad.numpy(), rtol=1e-5)


def test_parameter_registration_and_names():
    m = ConvNet(width=4)
    names = [n for n, _ in m.named_parameters()]
    assert names[0] == "features.0.weight" and "head.5.bias" in names
    assert len(names) == len(set(names))
    assert {n for n, _ in m.named_buffers()} >= {"features.1.running_mean", "features.1.running_var"}
    assert m.num_parameters() == sum(p.size for p in m.parameters())


def test_train_eval_toggles_dropout():
    m = nn.Sequential(nn.Dropout(0.5))
    x = Tensor(np.ones((100, 100)))
    assert (m(x).data == 0).mean() > 0.4
    m.eval()
    np.testing.assert_array_equal(m(x).data, x.data)
    assert all(not mod.training for mod in m.modules())


def test_dropout_gradient_is_the_mask():
    x = Tensor(np.ones((50, 50)), requires_grad=True)
    y = tg.F.dropout(x, 0.3, training=True)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, y.data)  # d(x*mask)/dx = mask (x == 1)
    assert set(np.unique(y.data).round(6)) <= {0.0, round(1 / 0.7, 6)}


def test_state_dict_roundtrip(tmp_path):
    m1, m2 = CharTransformer(11, 8, 16, 1, 2), CharTransformer(11, 8, 16, 1, 2)
    path = tmp_path / "ckpt.npz"
    tg.save(m1.state_dict(), path)
    m2.load_state_dict(tg.load(path))
    idx = np.random.default_rng(0).integers(0, 11, (2, 8))
    m1.eval(), m2.eval()
    np.testing.assert_array_equal(m1(idx).data, m2(idx).data)


def test_load_state_dict_strict():
    m = nn.Linear(2, 2)
    with pytest.raises(KeyError):
        m.load_state_dict({"weight": np.zeros((2, 2))})
    with pytest.raises(ValueError):
        m.load_state_dict({"weight": np.zeros((3, 2)), "bias": np.zeros(2)})
