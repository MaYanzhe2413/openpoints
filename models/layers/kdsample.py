import logging
import numpy as np
import torch
from typing import List

# CUDA FPS for leaf-internal sampling
try:
    from openpoints.cpp.pointnet2_batch import pointnet2_cuda
    CUDA_FPS_AVAILABLE = True
except ImportError:
    CUDA_FPS_AVAILABLE = False


def _fps_cuda(points: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    使用 CUDA 加速的 FPS，适用于单个点集（叶节点内）
    
    Args:
        points: (L, 3) - GPU 上的点云
        npoint: 采样点
    
    Returns:
        idx: (npoint,) - 采样索引
    """
    L = points.shape[0]
    if npoint >= L:
        return torch.arange(L, device=points.device, dtype=torch.long)
    
    if not points.is_cuda or not CUDA_FPS_AVAILABLE:
        return _fps_torch(points, npoint)
    
    # 转换为 batch 格式 (1, L, 3)
    pts = points.contiguous().unsqueeze(0).float()
    temp = torch.full((1, L), 1e10, device=points.device, dtype=torch.float32)
    output = torch.empty((1, npoint), device=points.device, dtype=torch.int32)
    
    pointnet2_cuda.furthest_point_sampling_wrapper(1, L, npoint, pts, temp, output)
    return output[0].long()


def _fps_torch(points: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    纯 PyTorch 实现的 FPS（当 CUDA 不可用时的后备方案）
    
    Args:
        points: (L, 3) - 点云
        npoint: 采样点数
    
    Returns:
        idx: (npoint,) - 采样索引
    """
    device = points.device
    L = points.shape[0]
    if npoint >= L:
        return torch.arange(L, device=device, dtype=torch.long)
    
    idx = torch.zeros(npoint, device=device, dtype=torch.long)
    dist = torch.full((L,), float('inf'), device=device)
    farthest = torch.zeros(1, device=device, dtype=torch.long)
    
    for i in range(npoint):
        idx[i] = farthest
        centroid = points[farthest, :].view(1, 3)
        d = torch.sum((points - centroid) ** 2, dim=1)
        dist = torch.minimum(dist, d)
        farthest = torch.argmax(dist)
    
    return idx



_kdtree_simple_sample_logged = False

def kdtree_simple_sample(xyz: torch.Tensor, npoint: int,
                              leaf_size: int = 32,
                              strategy: str = 'random',
                              proportional: bool = True,
                              axis_strategy: str = 'cycle') -> torch.Tensor:
    """
    更简KDTree叶节点采样：不执行叶内FPS，只做随机或均匀采样。

    Args:
        xyz: (B, N, 3)
        npoint: 目标采样点数
        leaf_size: 叶节点最大点数（用于递归划分）
        strategy: 'random' | 'uniform' | 'center_random' | 'quad_fps'
            random         : 叶内随机选择所需数量
            uniform        : 叶内按坐标排序后等间隔抽取
            center_random  : 先选叶中心附近的点，再用随机补足余下
            fps            : leaf-internal FPS (CUDA accelerated, GPU tree + leaf FPS)
            quad_fps       : 沿最大范围轴分4段，每段内FPS，拼接结果
        axis_strategy: 'cycle' | 'max_spread'
            cycle      : 沿 xyz 轴轮换 (depth % 3)，标准简化做法
            max_spread : 选当前节点范围最大的轴切分，更均衡
f.create_prefetch_pointmap()
            True  -> leaf_quota = round(len_leaf * npoint / N)
            False -> 所有叶平均分配 (基础 = npoint // L, 余数前 r 个 +1)

    Returns:
        idx: (B, npoint) 采样索引
    """
    global _kdtree_simple_sample_logged
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    if not _kdtree_simple_sample_logged:
        logging.info(f"[Sampler] kdtree_simple_sample | B={B}, N={N}, npoint={npoint}, leaf_size={leaf_size}, strategy={strategy}, proportional={proportional}, axis_strategy={axis_strategy} | recursive median-split ({axis_strategy}) + leaf {strategy} sampling")
        _kdtree_simple_sample_logged = True

    # 统一路径：全部使用 torch 实现，CPU/GPU 逻辑一致
    idx_all = torch.zeros(B, npoint, dtype=torch.long, device=device)
    coords = xyz  # (B,N,3)

    for b in range(B):
        # 递归划分得到叶节点
        init_idx = torch.arange(N, device=device, dtype=torch.long)
        leaf_nodes: List[torch.Tensor] = []
        stack = [(init_idx, 0)]  # (indices, depth)
        while stack:
            current, depth = stack.pop()
            if current.numel() <= leaf_size:
                leaf_nodes.append(current)
                continue
            if axis_strategy == 'max_spread':
                pts_b = coords[b, current]  # (M, 3)
                spread = pts_b.max(dim=0).values - pts_b.min(dim=0).values  # (3,)
                dim = int(spread.argmax().item())
            else:  # 'cycle' (default)
                dim = depth % 3
            pts = coords[b, current, dim]
            _, order = torch.sort(pts)
            median_pos = order.numel() // 2
            left_idx = current[order[:median_pos]]
            right_idx = current[order[median_pos:]]
            stack.append((right_idx, depth + 1))
            stack.append((left_idx, depth + 1))

        # 计算配额
        quotas = _compute_leaf_quotas_torch(leaf_nodes, N, npoint, proportional)

        sampled_list: List[int] = []
        for leaf, q in zip(leaf_nodes, quotas):
            Ls = leaf.numel()
            if q >= Ls:
                sampled_list.extend(leaf.tolist())
                continue

            if strategy == 'uniform':
                # 按最大范围轴排序后等间隔抽取
                pts_leaf = coords[b, leaf]  # (Ls, 3)
                ranges = (pts_leaf.max(dim=0).values - pts_leaf.min(dim=0).values)
                axis = int(torch.argmax(ranges))
                vals = pts_leaf[:, axis]
                _, order = torch.sort(vals)
                step = Ls / q
                chosen_local = [int(i * step) for i in range(q)]
                chosen = leaf[order[chosen_local]]

            elif strategy == 'quad_fps':
                # 沿最大范围轴分4段，每段内FPS
                pts_leaf = coords[b, leaf]  # (Ls, 3)
                ranges = (pts_leaf.max(dim=0).values - pts_leaf.min(dim=0).values)
                axis = int(torch.argmax(ranges))
                vals = pts_leaf[:, axis]
                _, order = torch.sort(vals)
                # Split into 4 contiguous bins
                s = order.numel()
                bsz = max(1, s // 4)
                bins = [order[i*bsz : (i+1)*bsz] for i in range(3)] + [order[3*bsz:]]
                parts = [leaf[idxs] for idxs in bins if idxs.numel() > 0]
                sizes = [p.numel() for p in parts]
                total = sum(sizes) if sizes else 0
                if not parts or total == 0:
                    perm = torch.randperm(Ls, device=device)[:q]
                    chosen = leaf[perm]
                else:
                    alloc = [max(1, int(round(sz * q / total))) for sz in sizes]
                    # rebalance to sum exactly q
                    diff = sum(alloc) - q
                    i = 0
                    while diff > 0 and any(a > 1 for a in alloc):
                        j = i % len(alloc)
                        if alloc[j] > 1:
                            alloc[j] -= 1
                            diff -= 1
                        i += 1
                    while diff < 0:
                        j = i % len(alloc)
                        if alloc[j] < sizes[j]:
                            alloc[j] += 1
                            diff += 1
                        i += 1
                    chosen_list = []
                    for part_idx, k_alloc in enumerate(alloc):
                        pidx = parts[part_idx]
                        if k_alloc <= 0:
                            continue
                        if k_alloc >= pidx.numel():
                            chosen_list.append(pidx)
                        else:
                            sub_pts = coords[b, pidx]
                            if sub_pts.is_cuda and CUDA_FPS_AVAILABLE:
                                try:
                                    sub_local = _fps_cuda(sub_pts, k_alloc)
                                except Exception:
                                    sub_local = _fps_torch(sub_pts, k_alloc)
                            else:
                                sub_local = _fps_torch(sub_pts, k_alloc)
                            chosen_list.append(pidx[sub_local])
                    chosen = torch.cat(chosen_list, dim=0) if chosen_list else torch.empty(0, dtype=torch.long, device=device)
                    if chosen.numel() > q:
                        chosen = chosen[:q]
                    elif chosen.numel() < q:
                        # top-up randomly from remaining in leaf
                        mask = torch.ones(Ls, device=device, dtype=torch.bool)
                        isin = torch.isin(leaf, chosen)
                        mask[isin.nonzero(as_tuple=False).flatten()] = False
                        remaining = leaf[mask]
                        need = q - chosen.numel()
                        if remaining.numel() > 0:
                            extra = remaining[torch.randperm(remaining.numel(), device=device)[:need]]
                            chosen = torch.cat([chosen, extra], dim=0)
                        else:
                            rep = chosen[torch.randint(0, max(1, chosen.numel()), (need,), device=device)] if chosen.numel() > 0 else leaf[torch.randperm(Ls, device=device)[:need]]
                            chosen = torch.cat([chosen, rep], dim=0)

            elif strategy == 'center_random':
                # 选最近质心点，其余随机
                pts_leaf = coords[b, leaf]  # (Ls, 3)
                centroid = pts_leaf.mean(dim=0, keepdim=True)
                d2 = ((pts_leaf - centroid) ** 2).sum(dim=1)
                center_idx_local = torch.argmin(d2)
                center_idx_global = leaf[center_idx_local]
                remain = torch.cat([leaf[:center_idx_local], leaf[center_idx_local + 1:]], dim=0)
                if q > 1:
                    perm = torch.randperm(remain.numel(), device=device)[: q - 1]
                    rest = remain[perm]
                    chosen = torch.cat([center_idx_global.unsqueeze(0), rest], dim=0)
                else:
                    chosen = center_idx_global.unsqueeze(0)

            elif strategy == 'fps':
                # leaf-internal FPS (CUDA accelerated)
                pts_leaf = coords[b, leaf]  # (Ls, 3)
                if pts_leaf.is_cuda and CUDA_FPS_AVAILABLE:
                    try:
                        local_idx = _fps_cuda(pts_leaf, q)
                    except Exception:
                        local_idx = _fps_torch(pts_leaf, q)
                else:
                    local_idx = _fps_torch(pts_leaf, q)
                chosen = leaf[local_idx]

            else:  # random
                perm = torch.randperm(Ls, device=device)[:q]
                chosen = leaf[perm]

            sampled_list.extend(chosen.tolist())

        # 调整数量
        if len(sampled_list) < npoint:
            remaining_mask = torch.ones(N, device=device, dtype=torch.bool)
            remaining_mask[sampled_list] = False
            remaining_idx = remaining_mask.nonzero(as_tuple=False).flatten()
            need = npoint - len(sampled_list)
            if remaining_idx.numel() >= need:
                extra = remaining_idx[torch.randperm(remaining_idx.numel(), device=device)[:need]].tolist()
            else:
                base = torch.as_tensor(sampled_list, device=device)
                repeat_extra = base[torch.randint(0, base.numel(), (need,), device=device)].tolist()
                extra = repeat_extra
            sampled_list.extend(extra)
        elif len(sampled_list) > npoint:
            perm = torch.randperm(len(sampled_list), device=device)[:npoint]
            sampled_list = [sampled_list[i] for i in perm.tolist()]

        idx_all[b] = torch.as_tensor(sampled_list[:npoint], device=device, dtype=torch.long)
    return idx_all


def _compute_leaf_quotas_torch(leaf_nodes: List[torch.Tensor], N: int, npoint: int, proportional: bool) -> List[int]:
    L = len(leaf_nodes)
    if proportional:
        raw = [(leaf.numel() * npoint) / N for leaf in leaf_nodes]
        quotas = [max(1, int(round(x))) for x in raw]
    else:
        base = npoint // L
        rem = npoint % L
        quotas = [base + (1 if i < rem else 0) for i in range(L)]
    sizes = [leaf.numel() for leaf in leaf_nodes]
    quotas = _rebalance_quotas(quotas, sizes, npoint, sizes_is_lengths=True)
    return quotas

def _rebalance_quotas(quotas, leaf_nodes, target, sizes_is_lengths=False):
    if sizes_is_lengths:
        sizes = leaf_nodes
    else:
        sizes = [len(leaf) for leaf in leaf_nodes]
    total_alloc = sum(quotas)
    if total_alloc > target:
        over = total_alloc - target
        adjustable = [i for i, q in enumerate(quotas) if q > 1]
        ri = 0
        while over > 0 and adjustable:
            i = adjustable[ri % len(adjustable)]
            if quotas[i] > 1:
                quotas[i] -= 1
                over -= 1
                if quotas[i] == 1:
                    adjustable.remove(i)
            ri += 1
    elif total_alloc < target:
        shortage = target - total_alloc
        order = sorted(range(len(quotas)), key=lambda i: sizes[i], reverse=True)
        oi = 0
        while shortage > 0 and order:
            i = order[oi % len(order)]
            if quotas[i] < sizes[i]:
                quotas[i] += 1
                shortage -= 1
            oi += 1
    return quotas
