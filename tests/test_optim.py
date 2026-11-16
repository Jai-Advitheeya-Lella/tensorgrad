"""Optimisers and schedulers vs. PyTorch: identical trajectories over many steps."""

import math

import numpy as np
import pytest

from tensorgrad import Tensor, optim

torch = pytest.importorskip("torch")

P0 = np.random.default_rng(0).standard_normal((4, 3))
C = np.random.default_rng(1).uniform(0.5, 2.0, (4, 3))


def loss_tg(p):
    return ((p * Tensor(C)) ** 2).sum() + p.sin().sum()


def loss_th(p):
    return ((p * torch.from_numpy(C)) ** 2).sum() + p.sin().sum()


@pytest.mark.parametrize(
    "tg_cls,th_cls,kwargs",
    [
        (optim.SGD, torch.optim.SGD, dict(lr=0.05)),
        (optim.SGD, torch.optim.SGD, dict(lr=0.05, momentum=0.9)),
        (optim.SGD, torch.optim.SGD, dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-2)),
        (optim.Adam, torch.optim.Adam, dict(lr=0.05)),
        (optim.Adam, torch.optim.Adam, dict(lr=0.05, betas=(0.8, 0.99), weight_decay=0.1)),
        (optim.AdamW, torch.optim.AdamW, dict(lr=0.05, weight_decay=0.1)),
    ],
    ids=["sgd", "sgd_momentum", "sgd_nesterov_wd", "adam", "adam_l2", "adamw"],
)
def test_optimizer_trajectory_matches_torch(tg_cls, th_cls, kwargs):
    p_tg = Tensor(P0.copy(), requires_grad=True)
    p_th = torch.tensor(P0.copy(), requires_grad=True)
    o_tg, o_th = tg_cls([p_tg], **kwargs), th_cls([p_th], **kwargs)
    for _ in range(25):
        o_tg.zero_grad()
        o_th.zero_grad()
        loss_tg(p_tg).backward()
        loss_th(p_th).backward()
        o_tg.step()
        o_th.step()
    np.testing.assert_allclose(p_tg.data, p_th.detach().numpy(), atol=1e-10, rtol=1e-8)


def test_param_groups_use_their_own_hyperparameters():
    a = Tensor(np.ones(2), requires_grad=True)
    b = Tensor(np.ones(2), requires_grad=True)
    opt = optim.SGD([{"params": [a]}, {"params": [b], "lr": 1.0}], lr=0.1)
    a.grad, b.grad = np.ones(2), np.ones(2)
    opt.step()
    np.testing.assert_allclose(a.data, 0.9)
    np.testing.assert_allclose(b.data, 0.0)


@pytest.mark.parametrize(
    "make_tg,make_th",
    [
        (lambda o: optim.StepLR(o, 7, 0.5), lambda o: torch.optim.lr_scheduler.StepLR(o, 7, 0.5)),
        (lambda o: optim.ExponentialLR(o, 0.9), lambda o: torch.optim.lr_scheduler.ExponentialLR(o, 0.9)),
        (lambda o: optim.CosineAnnealingLR(o, 20, 1e-3),
         lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, 20, 1e-3)),
        (lambda o: optim.LambdaLR(o, lambda s: 1 / (1 + s)),
         lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1 / (1 + s))),
    ],
    ids=["step", "exponential", "cosine", "lambda"],
)
def test_scheduler_matches_torch(make_tg, make_th):
    p = Tensor(np.zeros(1), requires_grad=True)
    q = torch.zeros(1, requires_grad=True)
    o_tg, o_th = optim.SGD([p], lr=0.1), torch.optim.SGD([q], lr=0.1)
    s_tg, s_th = make_tg(o_tg), make_th(o_th)
    for _ in range(20):
        assert math.isclose(o_tg.lr, o_th.param_groups[0]["lr"], rel_tol=1e-9, abs_tol=1e-12)
        o_th.step()
        s_tg.step()
        s_th.step()


def test_warmup_cosine_shape():
    p = Tensor(np.zeros(1), requires_grad=True)
    o = optim.SGD([p], lr=1.0)
    s = optim.WarmupCosineLR(o, warmup_steps=10, total_steps=110, min_ratio=0.1)
    lrs = []
    for _ in range(120):
        lrs.append(o.lr)
        s.step()
    assert np.isclose(lrs[0], 0.1) and np.isclose(lrs[9], 1.0)
    assert np.all(np.diff(lrs[10:110]) <= 1e-12) and np.isclose(lrs[-1], 0.1)
