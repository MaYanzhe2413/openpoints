import numpy as np
from sklearn.neighbors import KDTree
import torch
from typing import List, Tuple, Optional

# 导入 CUDA FPS
try:
    from openpoints.cpp.pointnet2_batch import pointnet2_cuda
    CUDA_FPS_AVAILABLE = True
except ImportError:
    CUDA_FPS_AVAILABLE = False


def kdtree_leaf_fps_sample(xyz: torch.Tensor, npoint: int, 
                          leaf_size: int = 32, fps_ratio: float = None) -> torch.Tensor:
    """
    遍历KDTree的每个叶节点，在每个叶节点内部进行FPS采样
    
    Args:
        xyz: (B, N, 3) - 输入点云
        npoint: int - 总采样点数
        leaf_size: int - KDTree叶节点最大大小
        fps_ratio: float - 每个叶节点内FPS采样的比例，None时自动计算为npoint/N
    
    Returns:
        idx: (B, npoint) - 采样索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    
    # 自动计算采样比例
    if fps_ratio is None:
        fps_ratio = npoint / N
    
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        points_np = xyz[b].detach().cpu().numpy()
        
        # 构建KDTree
        tree = KDTree(points_np, leaf_size=leaf_size)
        
        # 获取所有叶节点
        leaf_nodes = _extract_leaf_nodes(tree, points_np, leaf_size)
        
        # 在每个叶节点内进行FPS采样
        all_sampled_indices = []
        leaf_sample_counts = []
        
        for leaf_indices in leaf_nodes:
            if len(leaf_indices) == 0:
                continue
                
            # 计算这个叶节点应该采样多少个点
            leaf_npoint = max(1, int(len(leaf_indices) * fps_ratio))
            leaf_sample_counts.append(leaf_npoint)
            
            # 在叶节点内进行FPS采样
            leaf_points = points_np[leaf_indices]
            if len(leaf_indices) <= leaf_npoint:
                # 如果叶节点点数不够，全部选择
                sampled_in_leaf = list(range(len(leaf_indices)))
            else:
                # 在叶节点内进行FPS
                sampled_in_leaf = _fps_in_leaf(leaf_points, leaf_npoint)
            
            # 转换为原始索引
            for local_idx in sampled_in_leaf:
                all_sampled_indices.append(leaf_indices[local_idx])
        
        # 如果采样点数不够，从剩余点中补充
        if len(all_sampled_indices) < npoint:
            remaining_indices = [i for i in range(N) if i not in all_sampled_indices]
            additional_needed = npoint - len(all_sampled_indices)
            if remaining_indices:
                additional_samples = np.random.choice(
                    remaining_indices, 
                    min(additional_needed, len(remaining_indices)), 
                    replace=False
                )
                all_sampled_indices.extend(additional_samples)
        
        # 如果采样点数太多，随机选择
        if len(all_sampled_indices) > npoint:
            all_sampled_indices = np.random.choice(
                all_sampled_indices, npoint, replace=False
            )
        
        # 确保采样点数正确
        final_indices = all_sampled_indices[:npoint]
        idx[b] = torch.tensor(final_indices, dtype=torch.long, device=device)
    
    return idx

def kdtree_leaf_simple_sample(xyz: torch.Tensor, npoint: int,
                              leaf_size: int = 32,
                              strategy: str = 'random',
                              proportional: bool = True) -> torch.Tensor:
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
            quad_fps       : 沿最大范围轴分4段，每段内FPS，拼接结果
f.create_prefetch_pointmap()
            True  -> leaf_quota = round(len_leaf * npoint / N)
            False -> 所有叶平均分配 (基础 = npoint // L, 余数前 r 个 +1)

    Returns:
        idx: (B, npoint) 采样索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)

    # 统一路径：全部使用 torch 实现，CPU/GPU 逻辑一致
    idx_all = torch.zeros(B, npoint, dtype=torch.long, device=device)
    coords = xyz  # (B,N,3)

    for b in range(B):
        # 递归划分得到叶节点
        init_idx = torch.arange(N, device=device, dtype=torch.long)
        leaf_nodes: List[torch.Tensor] = []
        stack: List[torch.Tensor] = [init_idx]
        depth = 0
        while stack:
            current = stack.pop()
            if current.numel() <= leaf_size:
                leaf_nodes.append(current)
                continue
            dim = depth % 3
            pts = coords[b, current, dim]
            _, order = torch.sort(pts)
            median_pos = order.numel() // 2
            left_idx = current[order[:median_pos]]
            right_idx = current[order[median_pos:]]
            stack.append(right_idx)
            stack.append(left_idx)
            depth += 1

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


# ---- helper functions for kdtree_simple GPU/CPU paths ----
def _compute_leaf_quotas(leaf_nodes: List[List[int]], N: int, npoint: int, proportional: bool) -> List[int]:
    L = len(leaf_nodes)
    if proportional:
        raw = [len(leaf) * npoint / N for leaf in leaf_nodes]
        quotas = [max(1, int(round(x))) for x in raw]
    else:
        base = npoint // L
        rem = npoint % L
        quotas = [base + (1 if i < rem else 0) for i in range(L)]
    quotas = _rebalance_quotas(quotas, leaf_nodes, npoint)
    return quotas

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

# _sample_from_leaves_cpu 已废弃，统一使用 torch 路径


def kdtree_simple_sample(xyz: torch.Tensor, npoint: int,
                         leaf_size: int = 32,
                         strategy: str = 'random',
                         proportional: bool = True) -> torch.Tensor:
    """对外简单接口：叶节点随机/均匀采样，不使用叶内FPS。

    兼容现有 sampler 调用签名 (xyz, npoint)。
    可后续通过 partial 绑定 leaf_size / strategy。
    """
    return kdtree_leaf_simple_sample(xyz, npoint, leaf_size=leaf_size,
                                     strategy=strategy, proportional=proportional)


def _extract_leaf_nodes(tree: KDTree, points: np.ndarray, leaf_size: int) -> List[List[int]]:
    """
    提取KDTree的所有叶节点及其包含的点索引
    
    Args:
        tree: sklearn KDTree对象
        points: 原始点云数组 (N, 3)
        leaf_size: 叶节点最大大小
    
    Returns:
        leaf_nodes: List[List[int]] - 每个叶节点包含的点索引列表
    """
    leaf_nodes = []
    
    # 由于sklearn KDTree内部结构不易访问，我们用另一种方法：
    # 基于空间递归分割来模拟叶节点
    _recursive_spatial_split(points, list(range(len(points))), leaf_size, leaf_nodes)
    
    return leaf_nodes


def _recursive_spatial_split(points: np.ndarray, indices: List[int], 
                           leaf_size: int, leaf_nodes: List[List[int]], depth: int = 0) -> None:
    """
    递归空间分割，模拟KDTree的叶节点分割
    
    Args:
        points: 点云数组
        indices: 当前节点包含的点索引
        leaf_size: 叶节点最大大小
        leaf_nodes: 输出的叶节点列表
        depth: 当前递归深度
    """
    if len(indices) <= leaf_size:
        # 达到叶节点条件
        leaf_nodes.append(indices)
        return
    
    # 选择分割维度（轮流选择x, y, z）
    split_dim = depth % 3
    
    # 获取当前节点的点
    current_points = points[indices]
    
    # 按选定维度排序
    sorted_indices = sorted(indices, key=lambda i: points[i][split_dim])
    
    # 找到中位数分割点
    mid = len(sorted_indices) // 2
    
    # 递归分割左右子树
    left_indices = sorted_indices[:mid]
    right_indices = sorted_indices[mid:]
    
    _recursive_spatial_split(points, left_indices, leaf_size, leaf_nodes, depth + 1)
    _recursive_spatial_split(points, right_indices, leaf_size, leaf_nodes, depth + 1)


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


def _fps_in_leaf(points: np.ndarray, npoint: int) -> List[int]:
    """
    在叶节点内进FPS采样（优先使用 CUDA 加速）
    
    Args:
        points: 叶节点内的点 (M, 3)
        npoint: 要采样的点数
    
    Returns:
        sampled_indices: 采样点在叶节点内的索引
    """
    M = len(points)
    if M <= npoint:
        return list(range(M))
    
    # 尝试使用 GPU 加速
    if CUDA_FPS_AVAILABLE and torch.cuda.is_available():
        pts_cuda = torch.tensor(points, dtype=torch.float32, device='cuda')
        idx = _fps_cuda(pts_cuda, npoint)
        return idx.cpu().tolist()
    
    # CPU 回退：简化的FPS算法
    sampled_indices = []
    distances = np.full(M, np.inf)
    
    first_idx = 0
    sampled_indices.append(first_idx)
    
    for i in range(M):
        dist = np.linalg.norm(points[i] - points[first_idx])
        distances[i] = min(distances[i], dist)
    
    for _ in range(npoint - 1):
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)
        
        for i in range(M):
            dist = np.linalg.norm(points[i] - points[farthest_idx])
            distances[i] = min(distances[i], dist)
    
    return sampled_indices


def kdtree_adaptive_leaf_fps_sample(xyz: torch.Tensor, npoint: int, 
                                   leaf_size: int = 32, 
                                   density_adaptive: bool = True) -> torch.Tensor:
    """
    自适应的KDTree叶节点FPS采样 - 根据叶节点密度调整采样比例
    
    Args:
        xyz: (B, N, 3) - 输入点云
        npoint: int - 总采样点数
        leaf_size: int - KDTree叶节点最大大小
        density_adaptive: bool - 是否根据密度自适应调整采样比例
    
    Returns:
        idx: (B, npoint) - 采样索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        points_np = xyz[b].detach().cpu().numpy()
        tree = KDTree(points_np, leaf_size=leaf_size)
        
        # 获取所有叶节点
        leaf_nodes = _extract_leaf_nodes(tree, points_np, leaf_size)
        
        if density_adaptive:
            # 根据叶节点大小计算采样权重
            leaf_weights = [len(leaf) for leaf in leaf_nodes]
            total_weight = sum(leaf_weights)
            
            # 计算每个叶节点的采样数量
            leaf_sample_nums = []
            allocated_samples = 0
            
            for i, weight in enumerate(leaf_weights):
                if i == len(leaf_weights) - 1:
                    # 最后一个叶节点分配剩余的采样点
                    samples = npoint - allocated_samples
                else:
                    samples = int((weight / total_weight) * npoint)
                
                samples = min(samples, len(leaf_nodes[i]))  # 不超过叶节点大小
                leaf_sample_nums.append(samples)
                allocated_samples += samples
        else:
            # 均匀分配采样点
            samples_per_leaf = npoint // len(leaf_nodes)
            remainder = npoint % len(leaf_nodes)
            
            leaf_sample_nums = [samples_per_leaf] * len(leaf_nodes)
            for i in range(remainder):
                leaf_sample_nums[i] += 1
        
        # 在每个叶节点内进行FPS采样
        all_sampled_indices = []
        
        for leaf_indices, sample_num in zip(leaf_nodes, leaf_sample_nums):
            if sample_num == 0 or len(leaf_indices) == 0:
                continue
            
            leaf_points = points_np[leaf_indices]
            sampled_in_leaf = _fps_in_leaf(leaf_points, sample_num)
            
            # 转换为原始索引
            for local_idx in sampled_in_leaf:
                all_sampled_indices.append(leaf_indices[local_idx])
        
        # 确保采样点数正确
        if len(all_sampled_indices) != npoint:
            if len(all_sampled_indices) < npoint:
                # 补充采样点
                remaining = [i for i in range(N) if i not in all_sampled_indices]
                additional = np.random.choice(remaining, npoint - len(all_sampled_indices), replace=False)
                all_sampled_indices.extend(additional)
            else:
                # 随机选择
                all_sampled_indices = np.random.choice(all_sampled_indices, npoint, replace=False)
        
        idx[b] = torch.tensor(all_sampled_indices[:npoint], dtype=torch.long, device=device)
    
    return idx


# 简化接口，兼容PointNeXt
def kdtree_leaf_fps_simple(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """简化的KDTree叶节点FPS接口"""
    return kdtree_leaf_fps_sample(xyz, npoint)


if __name__ == "__main__":
    print("🧪 测试KDTree叶节点FPS采样...")
    
    # 创建测试数据
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建有明显聚类结构的点云
    cluster1 = np.random.normal([0, 0, 0], 0.3, (300, 3))
    cluster2 = np.random.normal([2, 0, 0], 0.3, (300, 3))
    cluster3 = np.random.normal([0, 2, 0], 0.3, (300, 3))
    cluster4 = np.random.normal([2, 2, 0], 0.3, (124, 3))  # 总共1024个点
    
    points = np.vstack([cluster1, cluster2, cluster3, cluster4])
    xyz = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # (1, 1024, 3)
    
    npoint = 256
    print(f"输入形状: {xyz.shape}")
    print(f"采样点数: {npoint}")
    
    # 测试1: 基本叶节点FPS采样
    print("\n1. 测试KDTree叶节点FPS采样...")
    idx1 = kdtree_leaf_fps_sample(xyz, npoint, leaf_size=32, fps_ratio=0.5)
    print(f"✅ 叶节点FPS采样结果: {idx1.shape}")
    
    # 测试2: 自适应叶节点FPS采样
    print("\n2. 测试自适应叶节点FPS采样...")
    idx2 = kdtree_adaptive_leaf_fps_sample(xyz, npoint, leaf_size=32, density_adaptive=True)
    print(f"✅ 自适应叶节点FPS采样结果: {idx2.shape}")
    
    # 测试3: 非自适应叶节点FPS采样
    print("\n3. 测试非自适应叶节点FPS采样...")
    idx3 = kdtree_adaptive_leaf_fps_sample(xyz, npoint, leaf_size=32, density_adaptive=False)
    print(f"✅ 非自适应叶节点FPS采样结果: {idx3.shape}")
    
    # 验证索引有效性
    for i, idx in enumerate([idx1, idx2, idx3], 1):
        assert torch.all(idx >= 0) and torch.all(idx < xyz.shape[1])
        assert idx.shape == (xyz.shape[0], npoint)
        print(f"✅ 测试{i}索引验证通过")
    
    # 分析采样结果
    print(f"\n📊 采样结果分析:")
    for i, (name, idx) in enumerate([
        ("叶节点FPS", idx1), 
        ("自适应叶节点FPS", idx2), 
        ("非自适应叶节点FPS", idx3)
    ], 1):
        sampled_points = xyz[0, idx[0]].numpy()
        
        # 计算每个聚类的采样点数
        cluster1_count = np.sum(np.linalg.norm(sampled_points - [0, 0, 0], axis=1) < 1.0)
        cluster2_count = np.sum(np.linalg.norm(sampled_points - [2, 0, 0], axis=1) < 1.0)
        cluster3_count = np.sum(np.linalg.norm(sampled_points - [0, 2, 0], axis=1) < 1.0)
        cluster4_count = np.sum(np.linalg.norm(sampled_points - [2, 2, 0], axis=1) < 1.0)
        
        print(f"{name}:")
        print(f"  聚类1采样: {cluster1_count}/300")
        print(f"  聚类2采样: {cluster2_count}/300") 
        print(f"  聚类3采样: {cluster3_count}/300")
        print(f"  聚类4采样: {cluster4_count}/124")
    
    print("✅ 所有测试通过!")