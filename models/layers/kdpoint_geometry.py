"""Runtime helpers for KDPoint's integer geometry contract."""

import math
from typing import Tuple

import torch


Q9_BITS = 9
Q9_LEVELS = (1 << Q9_BITS) - 1
WFU_FRAC_BITS = 7


def isotropic_q9_encode(points: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return q9 codes, fake-dequantized coordinates, and one step per cloud."""
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
        raise ValueError("points must have shape [B,N,3] with N > 0")
    if not points.is_floating_point():
        raise ValueError("source coordinates must be floating point")
    work = points.to(dtype=torch.float64)
    if not bool(torch.isfinite(work).all().detach().cpu().item()):
        raise ValueError("source coordinates contain non-finite values")
    origin = work.amin(dim=1, keepdim=True)
    extent = (work.amax(dim=1, keepdim=True) - origin).amax(
        dim=2, keepdim=True
    )
    if not bool((extent > 0).all().detach().cpu().item()):
        raise ValueError("each cloud must have a positive coordinate extent")
    step = extent / float(Q9_LEVELS)
    codes = torch.floor((work - origin) / step + 0.5)
    codes = codes.clamp(0, Q9_LEVELS).to(dtype=torch.int64)
    dequantized = (codes.to(torch.float64) * step + origin).to(points.dtype)
    return codes, dequantized, step


def q9_main_codes(codes: torch.Tensor) -> torch.Tensor:
    """Drop the encoder-only residual LSB and return the uint8 main path."""
    values = codes.to(dtype=torch.int64)
    return values >> 1


def strict_radius_sq(radius: float, step: torch.Tensor) -> torch.Tensor:
    """Convert physical ``d < radius`` to integer ``d2 <= threshold``."""
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    step_f64 = step.to(dtype=torch.float64)
    if not bool((step_f64 > 0).all().detach().cpu().item()):
        raise ValueError("coordinate step must be positive")
    return torch.ceil((radius / step_f64).square()).to(torch.int64) - 1


def strict_ball_query(ball_query_fn, radius: float, nsample: int,
                      support_q9: torch.Tensor, query_q9: torch.Tensor,
                      step: torch.Tensor) -> torch.Tensor:
    """Use the CUDA BQ kernel with a per-cloud radius inside each integer bin."""
    if support_q9.shape[0] != query_q9.shape[0]:
        raise ValueError("support/query batch sizes differ")
    thresholds = strict_radius_sq(radius, step).reshape(-1)
    outputs = []
    for batch_index in range(support_q9.shape[0]):
        threshold = int(thresholds[batch_index].detach().cpu().item())
        if threshold < 0:
            raise ValueError("radius is smaller than the q9 coordinate resolution")
        # Integer d2 values cannot fall in (threshold, threshold + 1).
        code_radius = math.sqrt(float(threshold) + 0.5)
        outputs.append(ball_query_fn(
            code_radius, nsample,
            support_q9[batch_index:batch_index + 1].contiguous(),
            query_q9[batch_index:batch_index + 1].contiguous(),
        ))
    return torch.cat(outputs, dim=0)


def _s32_tensor(value: torch.Tensor) -> torch.Tensor:
    unsigned = torch.bitwise_and(value.to(torch.int64), 0xffffffff)
    return torch.where(unsigned >= (1 << 31), unsigned - (1 << 32), unsigned)


def cordic_sqrt_distance_q15_tensor(distance_sq: torch.Tensor) -> torch.Tensor:
    """Vectorized bit-exact model of the RTL CORDIC square root."""
    values = distance_sq.to(dtype=torch.int64)
    if values.numel() == 0:
        return values
    if (int(values.min().detach().cpu().item()) < 0 or
            int(values.max().detach().cpu().item()) >= (1 << 18)):
        raise ValueError("WFU squared distances must fit 18 bits")

    input_data = values << 14
    highest = torch.zeros_like(values)
    for bit in range(32):
        highest = torch.where(
            torch.bitwise_and(input_data, 1 << bit) != 0,
            torch.full_like(highest, bit), highest,
        )
    shift_right = input_data >= 0x00020000
    shift_left = input_data < 0x00008000
    right_scale = highest - 16
    right_scale = right_scale + torch.bitwise_and(right_scale, 1)
    left_scale = 15 - highest
    left_scale = left_scale + torch.bitwise_and(left_scale, 1)
    scale = torch.where(
        shift_right, right_scale,
        torch.where(shift_left, left_scale, torch.zeros_like(highest)),
    )
    normalized = torch.where(
        shift_right, torch.bitwise_right_shift(input_data, scale),
        torch.where(shift_left, torch.bitwise_left_shift(input_data, scale),
                    input_data),
    )

    x = _s32_tensor(normalized + 0x00004000)
    y = _s32_tensor(normalized - 0x00004000)
    x = torch.where(x < 0, _s32_tensor(-x), x)
    for amount in (1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14):
        x_shift = torch.bitwise_right_shift(x, amount)
        y_shift = torch.bitwise_right_shift(y, amount)
        nonnegative = y >= 0
        next_x = torch.where(nonnegative, x - y_shift, x + y_shift)
        next_y = torch.where(nonnegative, y - x_shift, y + x_shift)
        x, y = _s32_tensor(next_x), _s32_tensor(next_y)

    gain = _s32_tensor(torch.bitwise_right_shift(x * 0x0001351e, 16))
    half_scale = torch.bitwise_right_shift(scale, 1)
    output = torch.where(
        shift_right, torch.bitwise_left_shift(gain, half_scale),
        torch.where(shift_left, torch.bitwise_right_shift(gain, half_scale), gain),
    )
    output = torch.bitwise_and(output, 0xffffffff)
    return torch.where(values == 0, torch.zeros_like(output), output)


def wfu_weights_exact_tensor(distances_sq: torch.Tensor,
                             frac_bits: int = WFU_FRAC_BITS) -> torch.Tensor:
    """Vectorized bit-exact WeightFinalizeUnit model; last dimension is K=3."""
    if distances_sq.shape[-1] != 3:
        raise ValueError("WFU requires a final dimension of three distances")
    if not 1 <= frac_bits <= 15:
        raise ValueError("WFU fractional width must be in [1,15]")
    distances = distances_sq.to(dtype=torch.int64)
    roots = torch.bitwise_and(
        cordic_sqrt_distance_q15_tensor(distances), (1 << 25) - 1
    )
    n0 = roots[..., 1] * roots[..., 2]
    n1 = roots[..., 0] * roots[..., 2]
    n2 = roots[..., 0] * roots[..., 1]
    denominator = n0 + n1 + n2
    any_zero = (distances == 0).any(dim=-1)
    safe_denominator = torch.where(
        any_zero, torch.ones_like(denominator), denominator
    )
    msb = torch.zeros_like(safe_denominator)
    for bit in range(52):
        msb = torch.where(
            torch.bitwise_and(safe_denominator, 1 << bit) != 0,
            torch.full_like(msb, bit), msb,
        )
    shift = torch.clamp(msb - 23, min=0)
    denominator_24 = torch.bitwise_right_shift(safe_denominator, shift)
    n1_24 = torch.bitwise_right_shift(n1, shift)
    n2_24 = torch.bitwise_right_shift(n2, shift)
    scale = 1 << frac_bits
    w1 = torch.div(n1_24 << frac_bits, denominator_24,
                   rounding_mode="floor")
    w2 = torch.div(n2_24 << frac_bits, denominator_24,
                   rounding_mode="floor")
    computed = torch.stack((scale - w1 - w2, w1, w2), dim=-1)

    zero = distances == 0
    zero_weights = torch.stack((
        zero[..., 0].to(torch.int64) * scale,
        ((~zero[..., 0]) & zero[..., 1]).to(torch.int64) * scale,
        ((~zero[..., 0]) & (~zero[..., 1]) & zero[..., 2]).to(torch.int64)
        * scale,
    ), dim=-1)
    return torch.where(any_zero.unsqueeze(-1), zero_weights, computed)


def _canonical_multiplier(ratio: float) -> Tuple[int, int]:
    mantissa, exponent = math.frexp(ratio)
    multiplier = math.floor(mantissa * (1 << 31) + 0.5)
    if multiplier == (1 << 31):
        multiplier >>= 1
        exponent += 1
    shift = 31 - exponent
    if not (1 << 30) <= multiplier < (1 << 31) or not 0 <= shift <= 63:
        raise ValueError("DpQuant ratio is outside the RTL multiplier range")
    while shift > 0 and multiplier % 2 == 0:
        multiplier //= 2
        shift -= 1
    return multiplier, shift


def dp_requantize_q9(delta_q9: torch.Tensor, step: torch.Tensor,
                     radius: float, target_scale: float,
                     target_zero_point: int, normalize_dp: bool) -> torch.Tensor:
    """Apply the runtime per-cloud DpQuant config and return fake-dequant data."""
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("target scale must be finite and positive")
    if not 0 <= target_zero_point <= 255:
        raise ValueError("target zero point must be uint8")
    delta = delta_q9.to(dtype=torch.int64)
    outputs = []
    for batch_index in range(delta.shape[0]):
        source_step = float(step.reshape(-1)[batch_index].detach().cpu().item())
        ratio = source_step / target_scale
        if normalize_dp:
            ratio /= float(radius)
        multiplier, shift = _canonical_multiplier(ratio)
        product = delta[batch_index] * multiplier
        if shift:
            product = product + (1 << (shift - 1))
        quantized = torch.bitwise_right_shift(product, shift)
        quantized = (quantized + target_zero_point).clamp(0, 255)
        outputs.append(quantized)
    codes = torch.stack(outputs, dim=0)
    return ((codes - target_zero_point).to(torch.float32) * target_scale)
