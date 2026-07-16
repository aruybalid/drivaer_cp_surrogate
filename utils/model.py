"""
Simple PointNet-style regressor for per-point Cp prediction from surface point cloud.
Input: [B, N, 6] (x,y,z, nx,ny,nz) normalized.
Output: [B, N, 1] Cp
Captures local features + global shape context via max-pooling.
Pure PyTorch, no extra geometric libs needed beyond torch.
"""

import torch
import torch.nn as nn

class PointNetCpRegressor(nn.Module):
    def __init__(self, input_dim=6, hidden=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        
        # Shared MLP for per-point local features
        layers = []
        prev = input_dim
        for h in [hidden, hidden, hidden]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        self.local_mlp = nn.Sequential(*layers)
        
        # Global feature from maxpool
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2)
        )
        
        # Final per-point regressor: concat(local, global) -> Cp
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden + hidden//2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        B, N, D = x.shape
        
        x_flat = x.view(B*N, D)
        
        # Local features
        local = self.local_mlp(x_flat)  # [B*N, hidden]
        
        local = local.view(B, N, -1)
        
        # Global feature (max over points)
        global_feat, _ = torch.max(local, dim=1)  # [B, hidden]
        
        global_feat = self.global_mlp(global_feat)  # [B, hidden//2]
        
        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden//2]
        
        # Concat and predict
        combined = torch.cat([local, global_feat], dim=-1)  # [B, N, hidden + hidden//2]
        
        combined_flat = combined.view(B*N, -1)
        
        out = self.final_mlp(combined_flat)  # [B*N, 1]
        
        result = out.view(B, N, 1)
        return result