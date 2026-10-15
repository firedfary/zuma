from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import ModelConfig

class SpatialFCNDQN(nn.Module):
    """
    Spatial Fully-Convolutional Dueling Q-Network for PopCap Zuma.
    Preserves 2D spatial layout to output a 32x24 spatial action heatmap (768 click points)
    alongside an auxiliary branch (2 actions: Right Click Swap, No-op).
    Total action output: 770 Q-values.
    """
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()
        in_channels = self.config.in_channels
        hidden_dim = self.config.hidden_dim
        grid_rows = self.config.grid_rows
        grid_cols = self.config.grid_cols
        num_aux = self.config.num_aux_actions

        # 1. Convolutional feature extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # 2. Spatial Advantage Branch (maps to 24x32 grid)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)  # (B, 1, grid_rows, grid_cols)
        )

        # 3. State-Value V(s) Branch (Dueling architecture)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.val_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # 4. Auxiliary Advantage Branch (Swap Ball, No-op)
        self.aux_fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_aux)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, 12, 180, 240) in [0, 1]
        Returns:
            q_values: Combined Q-values of shape (B, 770)
            spatial_map: 2D spatial Q-values of shape (B, 24, 32)
            aux_q: Auxiliary Q-values of shape (B, 2)
        """
        # Backbone feature extraction
        f = F.relu(self.bn1(self.conv1(x)))
        f = F.relu(self.bn2(self.conv2(f)))
        f = F.relu(self.bn3(self.conv3(f)))
        f = F.relu(self.bn4(self.conv4(f)))  # (B, 128, H', W')

        # Spatial Advantage Map
        # Interpolate features to target grid resolution (24, 32)
        f_grid = F.interpolate(
            f,
            size=(self.config.grid_rows, self.config.grid_cols),
            mode="bilinear",
            align_corners=False
        )
        adv_spatial = self.spatial_conv(f_grid)  # (B, 1, 24, 32)
        adv_spatial_flat = adv_spatial.view(x.size(0), -1)  # (B, 768)

        # Value Branch V(s)
        f_pooled = self.pool(f).view(x.size(0), -1)  # (B, 128*16)
        v = self.val_fc(f_pooled)  # (B, 1)

        # Auxiliary Advantage Branch A_aux(s)
        adv_aux = self.aux_fc(f_pooled)  # (B, 2)

        # Concatenate advantage streams: (B, 770)
        adv_all = torch.cat([adv_spatial_flat, adv_aux], dim=1)

        # Dueling formula: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = v + (adv_all - adv_all.mean(dim=1, keepdim=True))

        # Spatial map representation for heatmap visualization
        spatial_q = q_values[:, :self.config.grid_rows * self.config.grid_cols].view(
            x.size(0), self.config.grid_rows, self.config.grid_cols
        )
        aux_q = q_values[:, self.config.grid_rows * self.config.grid_cols:]

        return q_values, spatial_q, aux_q
