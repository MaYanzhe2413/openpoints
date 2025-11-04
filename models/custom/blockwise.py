# file: openpoints/models/custom/blockwise.py
import torch, torch.nn as nn

class BlockWiseTransfer(nn.Module):
    def __init__(self, block_size=0.4):
        super().__init__()
        self.block_size = block_size

    @staticmethod
    def _hash(x, s):
        """把坐标量化到 block，并 XOR 三轴得到 hash key"""
        key = torch.floor(x / s).int()
        return (key[:,0]*73856093 ^ key[:,1]*19349663 ^ key[:,2]*83492791)

    def forward(self, points_A, points_B):
        """
        points_A : 前一帧 (NA, 3+C)     已配准，前3列坐标，后C列特征
        points_B : 当帧   (NB, 3+C0)   前3列坐标，后C0列特征（可以为0）
        return: 
            diff_coords: 差分区域的点坐标 (N_diff, 3)
            matched_coords_features: 拼接区域的坐标+特征 (N_match, 3+C)
        """
        device = points_A.device
        
        # 分离坐标和特征
        xA, fA = points_A[:, :3], points_A[:, 3:]
        xB = points_B[:, :3]
        
        keyA, keyB = self._hash(xA, self.block_size), self._hash(xB, self.block_size)
        uniq = torch.unique(torch.cat([keyA, keyB]))
        
        # 存储差分区域和拼接区域的索引
        diff_indices = []
        matched_data = []

        for h in uniq.tolist():
            idxA = (keyA==h).nonzero(as_tuple=True)[0]
            idxB = (keyB==h).nonzero(as_tuple=True)[0]
            
            if len(idxB)==0: continue              # 跳过B中没有点的块
            if len(idxA)==0:                       # B中有点，A中没有点：全部归为差分区域
                diff_indices.append(idxB)
            elif len(idxB) > len(idxA):            # B点数大于A点数：差分区域
                diff_indices.append(idxB)
            else:                                  # B点数小于等于A点数：就近拼接
                dist = torch.cdist(xB[idxB][None], xA[idxA][None]).squeeze(0)
                nn_idx = dist.argmin(dim=1)
                # 用B的坐标和A的对应特征组合
                coords_b = xB[idxB]  # B的坐标
                features_a = fA[idxA[nn_idx]]  # A的对应特征
                matched_data.append(torch.cat([coords_b, features_a], dim=1))
        
        # 合并结果
        if diff_indices:
            diff_indices = torch.cat(diff_indices, dim=0)
            diff_coords = xB[diff_indices]
        else:
            diff_coords = torch.empty((0, 3), device=device)
            
        if matched_data:
            matched_coords_features = torch.cat(matched_data, dim=0)
        else:
            matched_coords_features = torch.empty((0, 3 + fA.shape[1]), device=device)
        
        return diff_coords, matched_coords_features
