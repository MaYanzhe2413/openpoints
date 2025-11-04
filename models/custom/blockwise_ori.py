# file: openpoints/models/custom/blockwise.py
import torch, torch.nn as nn
from openpoints.models.layers import furthest_point_sample
from openpoints.models.backbone.pointnetv2 import PointNetSAModuleMSG

class BlockWiseTransfer(nn.Module):
    def __init__(self, block_size=0.4, sa_cfg=None):
        super().__init__()
        self.block_size = block_size
        # 把一份 SA block 当子模块（只重算特征时用）
        self.sa = PointNetSAModuleMSG(**sa_cfg) if sa_cfg else None

    @staticmethod
    def _hash(x, s):
        """把坐标量化到 block，并 XOR 三轴得到 hash key"""
        key = torch.floor(x / s).int()
        return (key[:,0]*73856093 ^ key[:,1]*19349663 ^ key[:,2]*83492791)

    def forward(self, xA, fA, xB, fB):
        """
        xA/fA : 前一帧 (NA,3/C)     已配准
        xB/fB : 当帧   (NB,3/C0)   fB 初始可为 0
        return: fB_new (NB,C_out)
        """
        device = xA.device
        keyA, keyB = self._hash(xA, self.block_size), self._hash(xB, self.block_size)
        uniq = torch.unique(torch.cat([keyA, keyB]))
        fB_new = fB.clone()

        for h in uniq.tolist():
            idxA = (keyA==h).nonzero(as_tuple=True)[0]
            idxB = (keyB==h).nonzero(as_tuple=True)[0]
            if len(idxA)==0 or len(idxB)==0: continue
            if len(idxA) > len(idxB):              # 复制特征
                dist = torch.cdist(xB[idxB][None], xA[idxA][None]).squeeze(0)
                nn_idx = dist.argmin(dim=1)
                fB_new[idxB] = fA[idxA[nn_idx]]
            else:                                  # 重新 SA
                if self.sa is None:
                    continue
                xyz  = xB[idxB][None]
                feat = fB[idxB][None].T            # (1,C,N)
                _, f_upd = self.sa(xyz, feat)      # (1,C',N)
                fB_new[idxB] = f_upd.squeeze(0).T
        return fB_new
