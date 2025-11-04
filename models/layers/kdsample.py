import numpy as np
from sklearn.neighbors import KDTree
import torch
from typing import List, Tuple, Optional


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
    更简单的KDTree叶节点采样：不执行叶内FPS，只做随机或均匀采样。

    Args:
        xyz: (B, N, 3)
        npoint: 目标采样点数
        leaf_size: 叶节点最大点数（用于递归划分）
        strategy: 'random' | 'uniform'
            random  : 叶内随机选择所需数量
            uniform : 叶内按排序后等间隔抽取
        proportional: 是否按叶节点点数比例分配配额；
            True  -> leaf_quota = round(len_leaf * npoint / N)
            False -> 所有叶平均分配 (基础 = npoint // L, 余数前 r 个 +1)

    Returns:
        idx: (B, npoint) 采样索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)

    # 如果在 CPU 上，仍使用原 numpy 路径以减少重复逻辑
    if not xyz.is_cuda:
        xyz_np = xyz.detach().cpu().numpy()
        idx_all = torch.zeros(B, npoint, dtype=torch.long, device=device)
        for b in range(B):
            points = xyz_np[b]
            leaf_nodes: List[List[int]] = []
            _recursive_spatial_split(points, list(range(N)), leaf_size, leaf_nodes, depth=0)
            quotas = _compute_leaf_quotas(leaf_nodes, N, npoint, proportional)
            sampled = _sample_from_leaves_cpu(leaf_nodes, quotas, npoint, strategy, N)
            idx_all[b] = torch.as_tensor(sampled, device=device, dtype=torch.long)
        return idx_all

    # GPU 路径: 使用纯 torch 实现递归划分（仍在 Python 循环，但不转 numpy）
    idx_all = torch.zeros(B, npoint, dtype=torch.long, device=device)
    coords = xyz  # (B,N,3)
    for b in range(B):
        # 维护一个列表，每个元素是 (indices_tensor)
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
            sorted_vals, order = torch.sort(pts)
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
                step = Ls / q
                positions = (torch.arange(q, device=device, dtype=torch.float32) * step).long().clamp(max=Ls-1)
                chosen = leaf[positions]
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

def _sample_from_leaves_cpu(leaf_nodes: List[List[int]], quotas: List[int], npoint: int, strategy: str, N: int) -> List[int]:
    sampled_idx: List[int] = []
    for leaf, q in zip(leaf_nodes, quotas):
        if q <= 0:
            continue
        if q >= len(leaf):
            sampled_idx.extend(leaf)
            continue
        if strategy == 'uniform':
            leaf_sorted = sorted(leaf)
            step = len(leaf_sorted) / q
            chosen = [leaf_sorted[int(i * step)] for i in range(q)]
        else:
            chosen = np.random.choice(leaf, size=q, replace=False).tolist()
        sampled_idx.extend(chosen)
    if len(sampled_idx) < npoint:
        remaining = list(set(range(N)) - set(sampled_idx))
        need = npoint - len(sampled_idx)
        if len(remaining) >= need:
            sampled_idx.extend(np.random.choice(remaining, size=need, replace=False).tolist())
        else:
            extra = np.random.choice(sampled_idx, size=need, replace=True).tolist()
            sampled_idx.extend(extra)
    elif len(sampled_idx) > npoint:
        sampled_idx = np.random.choice(sampled_idx, size=npoint, replace=False).tolist()
    return sampled_idx[:npoint]


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


def _fps_in_leaf(points: np.ndarray, npoint: int) -> List[int]:
    """
    在叶节点内进行FPS采样
    
    Args:
        points: 叶节点内的点 (M, 3)
        npoint: 要采样的点数
    
    Returns:
        sampled_indices: 采样点在叶节点内的索引
    """
    M = len(points)
    if M <= npoint:
        return list(range(M))
    
    # 实现简化的FPS算法
    sampled_indices = []
    distances = np.full(M, np.inf)
    
    # 选择第一个点（随机或中心点）
    first_idx = 0  # 简单选择第一个点
    sampled_indices.append(first_idx)
    
    # 更新距离
    for i in range(M):
        dist = np.linalg.norm(points[i] - points[first_idx])
        distances[i] = min(distances[i], dist)
    
    # 迭代选择最远点
    for _ in range(npoint - 1):
        # 找到距离已采样点最远的点
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)
        
        # 更新距离
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