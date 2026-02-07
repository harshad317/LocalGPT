import os

import pytest
import torch

import nanochat.common as common


def _set_ddp_env(monkeypatch, *, local_rank: int, rank: int = 0, world_size: int = 8):
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


def test_compute_init_cuda_errors_on_invalid_local_rank(monkeypatch):
    _set_ddp_env(monkeypatch, local_rank=3)
    monkeypatch.delenv("NANOCHAT_OVERSUBSCRIBE_GPUS", raising=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(common.dist, "init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr(common.dist, "barrier", lambda: None)

    with pytest.raises(RuntimeError, match=r"Invalid LOCAL_RANK=3 for 2"):
        common.compute_init("cuda")


def test_compute_init_cuda_can_oversubscribe_with_env(monkeypatch):
    _set_ddp_env(monkeypatch, local_rank=3)
    monkeypatch.setenv("NANOCHAT_OVERSUBSCRIBE_GPUS", "1")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *_args, **_kwargs: None)
    seen = {"device": None}

    def _set_device(idx):
        seen["device"] = idx

    monkeypatch.setattr(torch.cuda, "set_device", _set_device)
    monkeypatch.setattr(common.dist, "init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr(common.dist, "barrier", lambda: None)

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = common.compute_init("cuda")
    assert ddp is True
    assert ddp_local_rank == 3
    assert ddp_world_size == 8
    assert str(device) == "cuda:1"
    assert seen["device"] == 1

