import os
import random
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..config import ModelConfig, TrainConfig
from ..model.spatial_fcn_dqn import SpatialFCNDQN

class ZumaDQNAgent:
    """
    Double DQN Agent with Spatial FCN architecture and Prioritized Replay support.
    """
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        train_config: Optional[TrainConfig] = None,
        device: Optional[str] = None
    ):
        self.m_config = model_config or ModelConfig()
        self.t_config = train_config or TrainConfig()
        
        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(self.t_config.device if torch.cuda.is_available() else "cpu")
            
        print(f"[agent] Initializing ZumaDQNAgent on device: {self.device}")

        # Online and Target Q-networks
        self.q_net = SpatialFCNDQN(self.m_config).to(self.device)
        self.target_net = SpatialFCNDQN(self.m_config).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.t_config.learning_rate)

        # Training counters
        self.train_step_count = 0
        self.epsilon = self.t_config.eps_start

    def get_epsilon(self) -> float:
        """
        Compute linearly decayed epsilon.
        """
        progress = min(1.0, self.train_step_count / max(1, self.t_config.eps_decay_steps))
        self.epsilon = self.t_config.eps_start + progress * (self.t_config.eps_end - self.t_config.eps_start)
        return self.epsilon

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        Args:
            obs: np.ndarray of shape (12, 180, 240) in [0, 1]
            evaluate: if True, greedily pick argmax Q
        Returns:
            action: integer in [0, 769]
        """
        eps = 0.0 if evaluate else self.get_epsilon()
        if random.random() < eps:
            total_actions = self.m_config.grid_rows * self.m_config.grid_cols + self.m_config.num_aux_actions
            return random.randint(0, total_actions - 1)

        self.q_net.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, 12, 180, 240)
            q_values, _, _ = self.q_net(x)
            action = int(q_values.argmax(dim=1).item())
        return action

    def update_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor]
    ) -> Tuple[float, np.ndarray]:
        """
        Execute one Double-DQN gradient update step.
        Returns:
            loss: float scalar loss
            td_errors: numpy array of TD errors for priority updating
        """
        self.q_net.train()
        states, actions, rewards, next_states, dones, indices, weights = batch

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        weights = weights.to(self.device).unsqueeze(1)

        # 1. Current Q(s, a)
        q_vals, _, _ = self.q_net(states)
        current_q = q_vals.gather(1, actions)

        # 2. Double DQN Target:
        # Action selection from online network: a* = argmax_a Q_online(s', a)
        with torch.no_grad():
            next_q_online, _, _ = self.q_net(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            # Action evaluation from target network: Q_target(s', a*)
            next_q_target, _, _ = self.target_net(next_states)
            max_next_q = next_q_target.gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.t_config.gamma * max_next_q

        # 3. TD error & weighted Smooth L1 Loss
        td_errors = (target_q - current_q).detach().squeeze(1).abs().cpu().numpy()
        loss = (weights * nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        # 4. Backward & Gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1

        # 5. Periodic hard target update
        if self.train_step_count % self.t_config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f"[agent] Target network synced at train step {self.train_step_count}")

        return float(loss.item()), td_errors

    def save_checkpoint(self, path: str, extra_info: Optional[Dict[str, Any]] = None):
        """
        Save model weights and optimizer state.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "train_step_count": self.train_step_count,
            "epsilon": self.epsilon,
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra_info": extra_info or {}
        }
        torch.save(checkpoint, path)
        print(f"[agent] Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """
        Load model weights and optimizer state.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step_count = checkpoint.get("train_step_count", 0)
        self.epsilon = checkpoint.get("epsilon", self.t_config.eps_start)
        print(f"[agent] Loaded checkpoint from {path} (train steps: {self.train_step_count})")
