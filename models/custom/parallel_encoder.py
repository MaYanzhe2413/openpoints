# file: openpoints/models/custom/parallel_encoder.py
import torch.nn as nn
from openpoints.utils import MODELS
from openpoints.models import build_model_from_cfg
from .blockwise import BlockWiseTransfer

@MODELS.register_module()
class ParallelEncoder(nn.Module):
    def __init__(self, base_args, block_args, fuse='concat', **kw):
        """
        base_args : PointNet2Encoder 的 cfg
        block_args: BlockWiseTransfer 的 cfg
        fuse      : 'concat' or 'add'
        """
        super().__init__()
        self.base   = build_model_from_cfg(base_args)          # PointNet2Encoder
        self.block  = BlockWiseTransfer(**block_args)
        self.fuse   = fuse

    @property
    def out_channels(self):
        if self.fuse == 'concat':
            return self.base.out_channels + self.block.sa.channel_list[-1][-1]
        return self.base.out_channels

    def forward_cls_feat(self, data):
        xyz, feat = data['pos'], data['x']       # (B,N,3), (B,C,N)
        B = xyz.shape[0]; assert B==1, "demo先支持单batch"

        base_feat = self.base.forward_cls_feat(data)           # (B,C1)
        # BlockWise path
        fB = feat.squeeze(0).T                                # (N,C0)
        fB_new = self.block(xyz.squeeze(0), fB, xyz.squeeze(0), fB)
        pool = fB_new.max(dim=0)[0][None]                     # (1,C2)

        if self.fuse == 'concat':
            return torch.cat([base_feat, pool], dim=1)         # (B,C1+C2)
        else:
            return base_feat + pool
