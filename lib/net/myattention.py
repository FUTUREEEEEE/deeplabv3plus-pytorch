import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels=cfg.MODEL_SHORTCUT_DIM, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(256, 256 // reduction_ratio),
            nn.ReLU(),
            nn.Linear(256 // reduction_ratio , 4*gate_channels)
            
            )
        self.expand=nn.Linear(gate_channels,4*gate_channels)
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            # elif pool_type=='lp':
            #     lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            #     channel_att_raw = self.mlp( lp_pool )
            # elif pool_type=='lse':
            #     # LSE pool only
            #     lse_pool = logsumexp_2d(x)
            #     channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(torch.zeros([cfg.TRAIN_BATCHES,cfg.MODEL_SHORTCUT_DIM*4,x.size(2),x.size(3)]))
        #scale=self.expand(scale)
        #
        return  scale


#hl.build_graph(net, torch.zeros([4, 48, 224, 224]))