import threading
from typing import Tuple, List, Optional
import numpy as np
import torch

class SumTree:
    """
    Binary SumTree data structure for O(log N) prioritized sampling.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data: object) -> int:
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(idx, p)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        return idx

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        if idx != 0:
            self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """
    Thread-safe Prioritized Experience Replay buffer (PER) with SumTree.
    """
    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 20000,
        eps: float = 1e-6
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.max_priority = 1.0
        self.frame_count = 0
        self.lock = threading.Lock()

    def __len__(self) -> int:
        with self.lock:
            return self.tree.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition (s, a, r, s', done) with maximal priority.
        """
        transition = (state, action, reward, next_state, done)
        with self.lock:
            p = (self.max_priority + self.eps) ** self.alpha
            self.tree.add(p, transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
        """
        Sample a mini-batch of prioritized transitions with importance sampling weights.
        Returns:
            states: (B, 12, 180, 240) float32
            actions: (B,) int64
            rewards: (B,) float32
            next_states: (B, 12, 180, 240) float32
            dones: (B,) float32
            indices: list of tree indices for priority update
            weights: (B,) importance sampling weights float32
        """
        with self.lock:
            if self.tree.size < batch_size:
                raise ValueError(f"Cannot sample {batch_size} transitions: buffer only contains {self.tree.size} items.")

            self.frame_count += 1
            beta = min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)

            batch = []
            indices = []
            priorities = []
            total_p = max(self.eps, self.tree.total())
            segment = total_p / batch_size

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                idx, p, data = self.tree.get(s)
                
                # Guard against uninitialized slots due to floating point drift
                attempts = 0
                while data is None and attempts < 20:
                    s = np.random.uniform(0, total_p)
                    idx, p, data = self.tree.get(s)
                    attempts += 1

                if data is None:
                    # Final fallback to any valid written data entry
                    valid_idx = np.random.randint(0, self.tree.size)
                    data = self.tree.data[valid_idx]
                    idx = valid_idx + self.capacity - 1
                    p = self.tree.tree[idx]

                indices.append(idx)
                priorities.append(max(self.eps, p))
                batch.append(data)

            # Compute Importance Sampling weights
            sampling_probs = np.array(priorities, dtype=np.float32) / (total_p + self.eps)
            weights = (self.tree.size * sampling_probs) ** (-beta)
            weights = weights / (weights.max() + self.eps)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32)
        weights_t = torch.tensor(weights, dtype=torch.float32)

        return states_t, actions_t, rewards_t, next_states_t, dones_t, indices, weights_t

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update TD-error priorities for sampled transitions with NaN/Inf sanitization.
        """
        with self.lock:
            for idx, p in zip(indices, priorities):
                if not (0 <= idx < len(self.tree.tree)):
                    continue
                p_clean = float(np.nan_to_num(p, nan=self.max_priority, posinf=self.max_priority, neginf=self.max_priority))
                p_abs = abs(p_clean)
                p_val = (p_abs + self.eps) ** self.alpha
                self.tree.update(idx, p_val)
                self.max_priority = max(self.max_priority, p_abs)
