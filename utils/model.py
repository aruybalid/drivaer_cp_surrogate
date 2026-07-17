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
    """Original PointNet with Global Max Pooling"""
    def __init__(self, input_dim=6, hidden=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        
        # Shared MLP for per-point local features
        layers = []
        prev = input_dim
        for h in [hidden] * num_layers:
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
        
        # Final per-point regressor
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden + hidden//2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        B, N, D = x.shape if x.dim() == 3 else (1, x.shape[0], x.shape[1])
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x_flat = x.view(B*N, D)
        
        # Local features
        local = self.local_mlp(x_flat)
        local = local.view(B, N, -1)
        
        # Global feature (max over points)
        global_feat, _ = torch.max(local, dim=1)
        global_feat = self.global_mlp(global_feat)
        
        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)
        
        # Concat and predict
        combined = torch.cat([local, global_feat], dim=-1)
        combined_flat = combined.view(B*N, -1)
        
        out = self.final_mlp(combined_flat)
        result = out.view(B, N, 1)
        return result


class MLPNoGlobalCpRegressor(nn.Module):
    """Plain MLP without global max pooling"""
    def __init__(self, input_dim=6, hidden=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        
        layers = []
        prev = input_dim
        for h in [hidden] * num_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, D = x.shape
        x_flat = x.view(B*N, D)
        out = self.mlp(x_flat)
        result = out.view(B, N, 1)
        return result


class MLPWithParamsCpRegressor(nn.Module):
    """MLP + 16 geometric parameters (no global max pooling)"""
    def __init__(self, input_dim=6, num_params=16, hidden=128):
        super().__init__()
        self.param_mlp = nn.Sequential(
            nn.Linear(num_params, 64),
            nn.ReLU(),
            nn.Linear(64, hidden),
            nn.ReLU(),
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, params):
        if x.dim() == 2: x = x.unsqueeze(0)
        if params.dim() == 1: params = params.unsqueeze(0)
        
        B, N, _ = x.shape
        point_features = self.point_mlp(x.view(B*N, -1)).view(B, N, -1)
        param_features = self.param_mlp(params).unsqueeze(1).expand(-1, N, -1)
        
        combined = torch.cat([point_features, param_features], dim=-1)
        out = self.final_mlp(combined.view(B*N, -1))
        result = out.view(B, N, 1)
        return result

class PointNetWithParamsCpRegressor(nn.Module):
    """PointNet + 16 geometric parameters (with global max pooling)"""
    def __init__(self, input_dim=6, num_params=16, hidden=128):
        super().__init__()
        self.hidden = hidden

        # Base PointNet (we only use it for local + global features)
        self.pointnet = PointNetCpRegressor(input_dim, hidden, num_layers=3)

        # MLP for the 16 geometric parameters
        self.param_mlp = nn.Sequential(
            nn.Linear(num_params, 64),
            nn.ReLU(),
            nn.Linear(64, hidden // 2),
            nn.ReLU(),
        )

        # Final regressor
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),   # ← Fixed: now expects 256
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, params):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)

        B, N, D = x.shape

        # Get local features from the base PointNet
        x_flat = x.view(B * N, D)
        local = self.pointnet.local_mlp(x_flat).view(B, N, -1)

        # Global feature via max pooling
        global_feat, _ = torch.max(local, dim=1)
        global_feat = self.pointnet.global_mlp(global_feat)   # [B, hidden//2]

        # Process geometric parameters
        param_feat = self.param_mlp(params)                   # [B, hidden//2]

        # Combine global + param features
        combined_global = torch.cat([global_feat, param_feat], dim=-1)   # [B, hidden]

        combined_global = combined_global.unsqueeze(1).expand(-1, N, -1)

        # Concatenate with local features
        combined = torch.cat([local, combined_global], dim=-1)           # [B, N, 256]

        # Final prediction
        out = self.final_mlp(combined.view(B * N, -1))
        result = out.view(B, N, 1)

        return result   # Always return shape [B, N, 1]