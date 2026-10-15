import threading
import time
from typing import Optional, Callable
import torch
import numpy as np

from ..config import ZumaConfig
from ..env.zuma_gym_env import ZumaGymEnv
from ..buffer.prioritized_replay import PrioritizedReplayBuffer
from ..model.spatial_fcn_dqn import SpatialFCNDQN

class ZumaActorWorker(threading.Thread):
    """
    Asynchronous Actor worker running at fixed 10 Hz decision frequency:
    - Non-blocking interaction loop with ZumaGymEnv
    - Pushes transitions into shared PrioritizedReplayBuffer
    - Periodically syncs network weights from Learner
    """
    def __init__(
        self,
        config: ZumaConfig,
        replay_buffer: PrioritizedReplayBuffer,
        stop_event: threading.Event,
        get_latest_weights_fn: Optional[Callable[[], Optional[dict]]] = None,
        get_epsilon_fn: Optional[Callable[[], float]] = None,
        name: str = "ZumaActor-10Hz"
    ):
        super().__init__(name=name)
        self.config = config
        self.replay_buffer = replay_buffer
        self.stop_event = stop_event
        self.get_latest_weights_fn = get_latest_weights_fn
        self.get_epsilon_fn = get_epsilon_fn

        self.env = ZumaGymEnv(config)
        self.local_model = SpatialFCNDQN(config.model).to("cpu")
        self.local_model.eval()

        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0

    def run(self):
        print(f"[actor] Starting Actor loop at {self.config.env.decision_hz} Hz...")
        obs, info = self.env.reset()

        while not self.stop_event.is_set():
            if self.env.controller.is_emergency_stopped():
                print("[actor] Emergency stop detected in actor worker.")
                self.stop_event.set()
                break

            # 1. Periodically sync weights from Learner (every 50 steps)
            if self.step_count % 50 == 0 and self.get_latest_weights_fn:
                latest_weights = self.get_latest_weights_fn()
                if latest_weights is not None:
                    try:
                        self.local_model.load_state_dict(latest_weights)
                    except Exception as e:
                        print(f"[actor] Weight sync error: {e}")

            # 2. Select action (epsilon-greedy)
            eps = self.get_epsilon_fn() if self.get_epsilon_fn else 0.1
            if np.random.rand() < eps:
                action = np.random.randint(0, self.config.env.total_actions)
            else:
                with torch.no_grad():
                    x = torch.from_numpy(obs).unsqueeze(0)  # CPU tensor
                    q_vals, _, _ = self.local_model(x)
                    action = int(q_vals.argmax(dim=1).item())

            # 3. Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated

            # 4. Push transition to replay buffer
            self.replay_buffer.push(obs, action, reward, next_obs, done)

            self.step_count += 1
            self.total_reward += reward

            if done:
                self.episode_count += 1
                print(
                    f"[actor] Episode {self.episode_count} finished | "
                    f"Steps: {step_info.get('episode_steps', 0)} | "
                    f"Score Gain: {step_info.get('episode_score_gain', 0)} | "
                    f"Total Reward: {self.total_reward:.2f}"
                )
                self.total_reward = 0.0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        print("[actor] Actor worker terminated cleanly.")
        self.env.close()
