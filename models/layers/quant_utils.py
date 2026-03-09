"""
Quantization utility modules for Eager Mode PTQ / QAT.

Provides nn.Module replacements for lambda-based pooling,
FloatFunctional wrappers for residual-add / concat,
and helper functions for Conv-BN fusion and FP32 pinning.
"""

import torch
import torch.nn as nn
import torch.quantization as quant


class MaxPool(nn.Module):
    """torch.max as nn.Module."""
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.max(
            x, dim=self.dim, keepdim=self.keepdim
        )[0]


class MeanPool(nn.Module):
    """torch.mean as nn.Module."""
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(
            x, dim=self.dim, keepdim=self.keepdim
        )


class SumPool(nn.Module):
    """torch.sum as nn.Module."""
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.sum(
            x, dim=self.dim, keepdim=self.keepdim
        )


def get_reduction_module(reduction='max', dim=-1, keepdim=False):
    """Return a pooling module given a string name."""
    reduction = reduction.lower()
    if reduction == 'max':
        return MaxPool(dim=dim, keepdim=keepdim)
    elif reduction in ('mean', 'avg'):
        return MeanPool(dim=dim, keepdim=keepdim)
    elif reduction == 'sum':
        return SumPool(dim=dim, keepdim=keepdim)
    else:
        raise ValueError(
            f'Unknown reduction: {reduction}'
        )


class QAdd(nn.Module):
    """Quantization-safe element-wise add."""
    def __init__(self):
        super().__init__()
        self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, x, y):
        return self.ff.add(x, y)


class QCat(nn.Module):
    """Quantization-safe torch.cat wrapper."""
    def __init__(self, dim=1):
        super().__init__()
        self.ff = torch.nn.quantized.FloatFunctional()
        self.dim = dim

    def forward(self, tensors):
        return self.ff.cat(tensors, dim=self.dim)


class QuantDeQuantBoundary(nn.Module):
    """Wraps inner module between QuantStub/DeQuantStub."""
    def __init__(self, inner):
        super().__init__()
        self.quant = quant.QuantStub()
        self.inner = inner
        self.dequant = quant.DeQuantStub()

    def forward(self, *args, **kwargs):
        args = tuple(
            self.quant(a) if isinstance(a, torch.Tensor)
            else a for a in args
        )
        out = self.inner(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return self.dequant(out)
        return out


def swap_custom_convs_to_standard(model):
    """Walk the whole model and swap custom Conv
    subclasses to standard nn.Conv for convert()
    compatibility. Call before torch.quantization.convert.
    """
    for name, child in model.named_modules():
        ctype = type(child)
        if (issubclass(ctype, nn.Conv1d)
                and ctype is not nn.Conv1d):
            child.__class__ = nn.Conv1d
        elif (issubclass(ctype, nn.Conv2d)
                and ctype is not nn.Conv2d):
            child.__class__ = nn.Conv2d
    return model


def swap_custom_convs_to_standard(model):
    """Walk the whole model and swap custom Conv
    subclasses to standard nn.Conv for convert()
    compatibility. Call before torch.quantization.convert.
    """
    for name, child in model.named_modules():
        ctype = type(child)
        if (issubclass(ctype, nn.Conv1d)
                and ctype is not nn.Conv1d):
            child.__class__ = nn.Conv1d
        elif (issubclass(ctype, nn.Conv2d)
                and ctype is not nn.Conv2d):
            child.__class__ = nn.Conv2d
    return model


def _swap_conv_to_standard(mod):
    """Replace custom Conv subclass with nn.Conv in-place.

    PyTorch fuse_modules uses exact type matching, so
    custom Conv1d/Conv2d subclasses fail. We swap them to
    standard nn.Conv and copy all state.
    """
    for key, child in mod.named_children():
        ctype = type(child)
        if (issubclass(ctype, nn.Conv1d)
                and ctype is not nn.Conv1d):
            child.__class__ = nn.Conv1d
        elif (issubclass(ctype, nn.Conv2d)
                and ctype is not nn.Conv2d):
            child.__class__ = nn.Conv2d


def fuse_convbn_modules(model):
    """Fuse Conv-BN(-ReLU) in-place.

    Handles custom Conv1d/Conv2d subclasses by temporarily
    swapping their __class__ to standard nn.Conv before fusion.
    """
    import torch.quantization as Q
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Sequential):
            ch = list(mod.children())
            try:
                if (len(ch) >= 3
                    and isinstance(ch[0], (nn.Conv1d, nn.Conv2d))
                    and isinstance(ch[1], (nn.BatchNorm1d, nn.BatchNorm2d))
                    and isinstance(ch[2], nn.ReLU)):
                    _swap_conv_to_standard(mod)
                    Q.fuse_modules(
                        mod, ['0', '1', '2'], inplace=True
                    )
                elif (len(ch) >= 2
                      and isinstance(ch[0], (nn.Conv1d, nn.Conv2d))
                      and isinstance(ch[1], (nn.BatchNorm1d, nn.BatchNorm2d))):
                    _swap_conv_to_standard(mod)
                    Q.fuse_modules(
                        mod, ['0', '1'], inplace=True
                    )
            except Exception:
                pass  # skip unfusable patterns
    return model


def disable_quantization_for_geometry(model):
    """Set qconfig=None on CUDA geometry modules."""
    from ..layers.group import (
        QueryAndGroup, GroupAll, KNNGroup
    )
    for name, mod in model.named_modules():
        if isinstance(mod, (QueryAndGroup, GroupAll, KNNGroup)):
            mod.qconfig = None
    return model


def print_fakequant_status(model, tag=""):
    """打印模型中所有 FakeQuantize 模块的状态。
    
    用于确认模型是否真正在 INT8 模拟模式下运行。
    调用时机：prepare_qat 之后、enable/disable_fake_quant 之后。
    """
    import logging
    logger = logging.getLogger(__name__)

    fq_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.quantization.FakeQuantize):
            fq_modules.append((name, mod))

    if not fq_modules:
        logger.info("[FakeQuant DEBUG%s] No FakeQuantize modules found — model is running in pure FP32",
                    f" {tag}" if tag else "")
        return

    enabled_count = 0
    disabled_count = 0
    observer_on = 0
    observer_off = 0

    for name, fq in fq_modules:
        if fq.fake_quant_enabled[0].item():
            enabled_count += 1
        else:
            disabled_count += 1
        if fq.observer_enabled[0].item():
            observer_on += 1
        else:
            observer_off += 1

    logger.info("[FakeQuant DEBUG%s] Total FakeQuantize modules: %d | "
                "fake_quant ON: %d, OFF: %d | observer ON: %d, OFF: %d",
                f" {tag}" if tag else "",
                len(fq_modules), enabled_count, disabled_count,
                observer_on, observer_off)

    # 打印前3个FakeQuantize的详细信息作为示例
    for name, fq in fq_modules[:3]:
        fq_on = bool(fq.fake_quant_enabled[0].item())
        obs_on = bool(fq.observer_enabled[0].item())
        qmin, qmax = fq.quant_min, fq.quant_max
        scale = fq.scale.item() if fq.scale.numel() == 1 else f"tensor({fq.scale.shape})"
        zp = fq.zero_point.item() if fq.zero_point.numel() == 1 else f"tensor({fq.zero_point.shape})"
        logger.info("  [sample] %s | fake_quant=%s, observer=%s | "
                    "qrange=[%d,%d], scale=%s, zero_point=%s",
                    name, fq_on, obs_on, qmin, qmax, scale, zp)
    if len(fq_modules) > 3:
        logger.info("  ... and %d more FakeQuantize modules", len(fq_modules) - 3)
