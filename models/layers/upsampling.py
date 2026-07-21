from typing import List, Tuple
from torch.autograd import Function

import torch
import torch.nn as nn

from openpoints.cpp.pointnet2_batch import pointnet2_cuda
from openpoints.models.layers import create_convblock1d
from .kdpoint_geometry import q9_main_codes, wfu_weights_exact_tensor


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2_cuda.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2_cuda.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.zeros([B, c, m], device='cuda', requires_grad=True)
        grad_out_data = grad_out.data.contiguous()

        pointnet2_cuda.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


def _fake_quant_xyz(p, nbits):
    """Fake-quantize coordinates (per-sample, per-axis uniform).
    nbits<=0 : no-op (FP32)
    nbits=-1 : FP16
    nbits>=1 : n-bit fixed-point uniform quant (e.g. 8/12/16 = int8/int12/int16)"""
    if not nbits or nbits == 0:
        return p
    if nbits == -1:
        return p.half().float()
    if nbits < 0:
        return p
    levels = float(2 ** nbits - 1)
    p_min = p.amin(dim=1, keepdim=True)
    p_max = p.amax(dim=1, keepdim=True)
    scale = (p_max - p_min).clamp(min=1e-6) / levels
    return torch.round((p - p_min) / scale) * scale + p_min


def three_interpolation(unknown_xyz, known_xyz, know_feat,
                        knn_nbits=0, weight_nbits=0, topk=0,
                        hardware_exact=False):
    """
    input: known_xyz: (m, 3), unknown_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)

    Coordinate-quantization ablation knobs (all default 0 = original FP32 behaviour):
        knn_nbits   : coord precision for selecting the 3-NN indices
        weight_nbits: coord precision for computing the interpolation weights
        topk        : if >3, select topk candidates with knn_nbits coords,
                      then refine to top3 with weight_nbits coords
    """
    if hardware_exact:
        unknown_main = q9_main_codes(unknown_xyz).to(
            dtype=unknown_xyz.dtype
        ).contiguous()
        known_main = q9_main_codes(known_xyz).to(
            dtype=known_xyz.dtype
        ).contiguous()
        _, idx = three_nn(unknown_main, known_main)
        batch_size, n_fine, _ = unknown_main.shape
        neighbors = torch.gather(
            known_main.unsqueeze(1).expand(batch_size, n_fine, -1, -1), 2,
            idx.long().unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        delta = neighbors.to(torch.int64) - unknown_main.to(
            torch.int64
        ).unsqueeze(2)
        distances_sq = delta.square().sum(dim=-1)
        weight_codes = wfu_weights_exact_tensor(distances_sq)
        weight = (weight_codes.to(dtype=know_feat.dtype) / 128.0).contiguous()
        return three_interpolate(know_feat, idx.contiguous(), weight)

    # original fast path
    if knn_nbits <= 0 and weight_nbits <= 0 and topk <= 0:
        dist, idx = three_nn(unknown_xyz, known_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        return three_interpolate(know_feat, idx, weight)

    # ablation path
    B, N, _ = unknown_xyz.shape
    xyz_knn = _fake_quant_xyz(unknown_xyz, knn_nbits)
    kxyz_knn = _fake_quant_xyz(known_xyz, knn_nbits)
    xyz_w = _fake_quant_xyz(unknown_xyz, weight_nbits)
    kxyz_w = _fake_quant_xyz(known_xyz, weight_nbits)

    if topk and topk > 3:
        # int8 (knn-precision) topk candidates + weight-precision refine.
        # chunk over fine points to avoid materializing the full (N,M) cdist.
        K = min(int(topk), kxyz_knn.shape[1])
        chunk = 4096
        idx_parts, dist_parts = [], []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            c = e - s
            cd = torch.cdist(xyz_knn[:, s:e], kxyz_knn)              # (B,c,M)
            _, cand_idx = cd.topk(K, dim=2, largest=False)           # (B,c,K)
            # refine: recompute candidate distances with weight-precision coords
            cand_kxyz = torch.gather(
                kxyz_w.unsqueeze(1).expand(B, c, -1, -1), 2,
                cand_idx.unsqueeze(-1).expand(-1, -1, -1, 3))        # (B,c,K,3)
            refine_d = torch.norm(
                cand_kxyz - xyz_w[:, s:e].unsqueeze(2), dim=-1)      # (B,c,K)
            top3_d, top3_local = refine_d.topk(3, dim=2, largest=False)
            idx_parts.append(torch.gather(cand_idx, 2, top3_local))
            dist_parts.append(top3_d)
        idx = torch.cat(idx_parts, dim=1).int().contiguous()        # (B,N,3)
        dist = torch.cat(dist_parts, dim=1)                         # (B,N,3)
    else:
        # 3-NN idx from knn-precision coords
        _, idx = three_nn(xyz_knn.contiguous(), kxyz_knn.contiguous())  # (B,N,3) int
        # recompute neighbour distances with weight-precision coords
        nb_kxyz = torch.gather(
            kxyz_w.unsqueeze(1).expand(B, N, -1, -1), 2,
            idx.long().unsqueeze(-1).expand(-1, -1, -1, 3))          # (B,N,3,3)
        dist = torch.norm(nb_kxyz - xyz_w.unsqueeze(2), dim=-1)      # (B,N,3)
        idx = idx.int().contiguous()

    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = (dist_recip / norm).contiguous()
    return three_interpolate(know_feat, idx, weight)


if __name__ == "__main__":
    pass
