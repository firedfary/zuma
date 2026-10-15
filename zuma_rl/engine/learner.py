import threading
import time
from typing import Optional, Dict
import copy
import numpy as np

from ..config import ZumaConfig
from ..buffer.prioritized_replay import PrioritizedReplayBuffer
from ..agent.dqn_agent import ZumaDQNAgent

class ZumaLearnerWorker(threading.Thread):
    """
    Asynchronous Learner worker running on GPU:
    - Independent gradient update loop
    - Samples batches from PrioritizedReplayBuffer
    - Executes Double DQN training steps and updates TD priorities
    - Thread-safely publishes latest weights for Actor
    """
    def __init__(
        self,
        config: ZumaConfig,
        replay_buffer: PrioritizedReplayBuffer,
        stop_event: threading.Event,
        name: str = "ZumaLearner-GPU"
    ):
        super().__init__(name=name)
        self.config = config
        self.t_config = config.train
        self.replay_buffer = replay_buffer
        self.stop_event = stop_event

        self.agent = ZumaDQNAgent(
            model_config=config.model,
            train_config=config.train,
            device=self.t_config.device
        )

        # Thread-safe storage for weights to pass to Actor
        self._latest_weights_lock = threading.Lock()
        self._latest_weights: Optional[Dict[str, object]] = None
        self._sync_weights_to_buffer()

        self.total_train_steps = 0
        self.recent_losses = []

    def _sync_weights_to_buffer(self):
        cpu_state = {k: v.cpu().clone() for k, v in self.agent.q_net.state_dict().items()}
        with self._latest_weights_lock:
            self._latest_weights = cpu_state

    def get_latest_weights(self) -> Optional[Dict[str, object]]:
        with self._latest_weights_lock:
            return copy.deepcopy(self._latest_weights)

    def get_current_epsilon(self) -> float:
        return self.agent.get_epsilon()

    def run(self):
        print(f"[learner] Starting Learner worker on device {self.agent.device}...")
        start_time = time.time()
        last_log_time = time.time()

        while not self.stop_event.is_set():
            if self.total_train_steps >= self.t_config.max_train_steps:
                print(f"[learner] Reached max training steps ({self.t_config.max_train_steps}).")
                self.stop_event.set()
                break

            # 1. Check if buffer has enough warmup transitions
            if len(self.replay_buffer) < self.t_config.min_replay_size:
                time.sleep(0.1)
                continue

            # 2. Sample prioritized batch
            batch = self.replay_buffer.sample(self.t_config.batch_size)
            indices = batch[5]

            # 3. Double DQN gradient step
            loss, td_errors = self.agent.update_step(batch)
            self.recent_losses.append(loss)
            if len(self.recent_losses) > 100:
                self.recent_losses.pop(0)

            # 4. Update PER transition priorities
            self.replay_buffer.update_priorities(indices, td_errors)

            self.total_train_steps += 1

            # 5. Periodically publish latest weights (every 20 steps)
            if self.total_train_steps % 20 == 0:
                self._sync_weights_to_buffer()

            # 6. Logging
            if self.total_train_steps % self.t_config.log_interval_steps == 0:
                now = time.time()
                elapsed = now - last_log_time
                steps_per_sec = self.t_config.log_interval_steps / max(1e-5, elapsed)
                last_log_time = now
                avg_loss = float(np.mean(self.recent_losses))
                eps = self.agent.get_epsilon()
                print(
                    f"[learner] Step {self.total_train_steps:6d} | "
                    f"Buffer: {len(self.replay_buffer):5d} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Eps: {eps:.3f} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

            # 7. Checkpoint saving
            if self.total_train_steps % self.t_config.checkpoint_interval_steps == 0:
                ckpt_path = f"{self.t_config.checkpoint_dir}/zuma_dqn_step_{self.total_train_steps}.pt"
                self.agent.save_checkpoint(ckpt_path, {"buffer_size": len(self.replay_buffer)})

        # Final checkpoint save
        final_ckpt = f"{self.t_config.checkpoint_dir}/zuma_dqn_final.pt"
        self.agent.save_checkpoint(final_ckpt, {"buffer_size": len(self.replay_buffer)})
        print("[learner] Learner worker terminated cleanly.")
