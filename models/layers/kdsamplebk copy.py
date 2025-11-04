import numpy as np
from sklearn.neighbors import KDTree
import torch
from typing import Tuple, Optional

def kdtree_leaf_fps_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Args:
        xyz: (B, N, 3) - 输入点云，N个原始点
        npoint: int - 要采样的点数，必须 <= N
    
    Returns:
        idx: (B, npoint) - 采样得到的npoint个点的索引
    """
    B, N, C = xyz.shape
    npoint = min(npoint, N)  # 确保不超过原始点数
    ratio = npoint / N

    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)

    for b in range(B):
        # 将点云数据转换为numpy数组以构建KDTree
        points = xyz[b].cpu().numpy()  # (N, 3)
        tree = KDTree(points)

        # 计算每个点的采样概率
        distances, _ = tree.query(points, k=2)  # 最近邻距离
        prob = distances[:, 1]  # 使用第二近邻距离作为采样概率
        prob = prob / prob.sum()  # 归一化

        # 根据概率进行采样
        sampled_indices = np.random.choice(N, npoint, replace=False, p=prob)
        idx[b] = torch.from_numpy(sampled_indices).to(device)
    
    # 你的采样逻辑...
    # 最终返回 npoint 个选中的点的索引
    return idx  # 形状: (B, npoint)

def kdtree_fps_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    基于KDTree的最远点采样 (类似FPS，但使用KDTree加速)
    
    Args:
        xyz: (B, N, 3) - 输入点云，N个原始点
        npoint: int - 要采样的点数，必须 <= N
    
    Returns:
        idx: (B, npoint) - 采样得到的npoint个点的索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)  # 确保不超过原始点数
    
    # 初始化输出
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        # 转换为numpy (KDTree需要numpy数组)
        points_np = xyz[b].detach().cpu().numpy()
        
        # 构建KDTree
        tree = KDTree(points_np)
        
        # 采样算法：类似FPS但使用KDTree加速
        sampled_indices = []
        
        # 1. 选择第一个点（随机选择或选择中心点）
        if npoint > 0:
            # 选择最接近点云中心的点
            center = np.mean(points_np, axis=0)
            _, first_idx = tree.query([center], k=1)
            sampled_indices.append(first_idx[0][0])
        
        # 2. 迭代选择剩余点
        for i in range(1, npoint):
            max_min_distance = -1
            best_candidate = -1
            
            # 对于每个未选择的点，计算到最近已选择点的距离
            for candidate in range(N):
                if candidate in sampled_indices:
                    continue
                
                # 使用KDTree快速找到候选点到已选择点的最近距离
                candidate_point = points_np[candidate:candidate+1]
                sampled_points = points_np[sampled_indices]
                
                # 构建已选择点的临时KDTree
                if len(sampled_indices) > 1:
                    sampled_tree = KDTree(sampled_points)
                    min_distance, _ = sampled_tree.query(candidate_point, k=1)
                    min_distance = min_distance[0][0]
                else:
                    # 只有一个已选择点时直接计算距离
                    min_distance = np.linalg.norm(candidate_point[0] - sampled_points[0])
                
                # 更新最远点
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate != -1:
                sampled_indices.append(best_candidate)
        
        # 转换回tensor
        idx[b] = torch.tensor(sampled_indices, dtype=torch.long, device=device)
    
    return idx


def kdtree_uniform_sample(xyz: torch.Tensor, npoint: int, 
                         grid_resolution: float = None) -> torch.Tensor:
    """
    基于KDTree的均匀网格采样
    
    Args:
        xyz: (B, N, 3) - 输入点云
        npoint: int - 采样点数
        grid_resolution: float - 网格分辨率，None时自动计算
    
    Returns:
        idx: (B, npoint) - 采样索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        points_np = xyz[b].detach().cpu().numpy()
        tree = KDTree(points_np)
        
        # 计算点云边界
        min_coords = np.min(points_np, axis=0)
        max_coords = np.max(points_np, axis=0)
        bbox_size = max_coords - min_coords
        
        # 自动计算网格分辨率
        if grid_resolution is None:
            grid_resolution = np.max(bbox_size) / (npoint ** (1/3))
        
        # 生成网格中心点
        grid_points = []
        x_steps = int(bbox_size[0] / grid_resolution) + 1
        y_steps = int(bbox_size[1] / grid_resolution) + 1
        z_steps = int(bbox_size[2] / grid_resolution) + 1
        
        for i in range(x_steps):
            for j in range(y_steps):
                for k in range(z_steps):
                    grid_center = min_coords + np.array([i, j, k]) * grid_resolution
                    if len(grid_points) < npoint * 2:  # 生成足够的候选点
                        grid_points.append(grid_center)
        
        # 为每个网格点找最近的实际点
        sampled_indices = []
        used_indices = set()
        
        for grid_center in grid_points:
            if len(sampled_indices) >= npoint:
                break
            
            # 找到最近的实际点
            distances, indices = tree.query([grid_center], k=5)  # 找5个最近邻
            
            for idx_candidate in indices[0]:
                if idx_candidate not in used_indices:
                    sampled_indices.append(idx_candidate)
                    used_indices.add(idx_candidate)
                    break
        
        # 如果网格采样不够，随机补充
        while len(sampled_indices) < npoint:
            remaining = [i for i in range(N) if i not in used_indices]
            if not remaining:
                break
            random_idx = np.random.choice(remaining)
            sampled_indices.append(random_idx)
            used_indices.add(random_idx)
        
        # 截断到所需数量
        sampled_indices = sampled_indices[:npoint]
        idx[b] = torch.tensor(sampled_indices, dtype=torch.long, device=device)
    
    return idx


def kdtree_density_sample(xyz: torch.Tensor, npoint: int, 
                         radius: float = 0.1) -> torch.Tensor:
    """
    基于密度的KDTree采样 - 在密度高的区域采样更多点
    
    Args:
        xyz: (B, N, 3) - 输入点云
        npoint: int - 采样点数
        radius: float - 密度计算半径
    
    Returns:
        idx: (B, npoint) - 采样索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    
    for b in range(B):
        points_np = xyz[b].detach().cpu().numpy()
        tree = KDTree(points_np)
        
        # 1. 计算每个点的密度 (半径内的邻居数量)
        densities = []
        for i in range(N):
            neighbors = tree.query_radius([points_np[i]], r=radius)[0]
            densities.append(len(neighbors))
        
        densities = np.array(densities)
        
        # 2. 根据密度加权采样
        # 密度越高，被选中的概率越大
        probabilities = densities / np.sum(densities)
        
        # 3. 使用加权随机采样
        sampled_indices = np.random.choice(
            N, size=npoint, replace=False, p=probabilities
        )
        
        idx[b] = torch.tensor(sampled_indices, dtype=torch.long, device=device)
    
    return idx


# 简化接口，兼容PointNeXt
def kdtree_sample_simple(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """简化的KDTree采样接口，兼容PointNeXt的FPS接口"""
    return kdtree_sample(xyz, npoint)


if __name__ == "__main__":
    print("🧪 测试KDTree采样算法...")
    
    # 创建测试数据
    torch.manual_seed(42)
    xyz = torch.randn(2, 1024, 3)
    npoint = 256
    
    print(f"输入形状: {xyz.shape}")
    print(f"采样点数: {npoint}")
    
    # 测试1: 基本KDTree采样 (类似FPS)
    print("\n1. 测试KDTree FPS采样...")
    idx1 = kdtree_sample(xyz, npoint)
    print(f"✅ KDTree FPS采样结果: {idx1.shape}")
    
    # 测试2: 均匀网格采样
    print("\n2. 测试均匀网格采样...")
    idx2 = kdtree_uniform_sample(xyz, npoint)
    print(f"✅ 均匀网格采样结果: {idx2.shape}")
    
    # 测试3: 密度加权采样
    print("\n3. 测试密度加权采样...")
    idx3 = kdtree_density_sample(xyz, npoint)
    print(f"✅ 密度加权采样结果: {idx3.shape}")
    
    # 验证索引有效性
    for i, idx in enumerate([idx1, idx2, idx3], 1):
        assert torch.all(idx >= 0) and torch.all(idx < xyz.shape[1])
        assert idx.shape == (xyz.shape[0], npoint)
        print(f"✅ 测试{i}索引验证通过")
    
    # 性能对比测试
    print("\n🚀 性能测试...")
    import time
    
    # 大数据测试
    large_xyz = torch.randn(4, 4096, 3)
    large_npoint = 1024
    
    start_time = time.time()
    _ = kdtree_sample(large_xyz, large_npoint)
    kdtree_time = time.time() - start_time
    
    print(f"KDTree采样时间: {kdtree_time:.4f}s (输入: {large_xyz.shape})")
    print("✅ 所有测试通过!")