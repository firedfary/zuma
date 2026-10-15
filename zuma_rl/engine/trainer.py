import threading
import time
import signal
from typing import Optional

from ..config import ZumaConfig
from ..buffer.prioritized_replay import PrioritizedReplayBuffer
from .actor import ZumaActorWorker
from .learner import ZumaLearnerWorker

class ZumaTrainer:
    """
    Main training orchestrator for Zuma Reinforcement Learning system.
    Coordinates concurrent Actor (10 Hz interaction) and Learner (GPU updates).
    """
    def __init__(self, config: Optional[ZumaConfig] = None):
        self.config = config or ZumaConfig()
        self.stop_event = threading.Event()
        
        # Shared prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.train.replay_capacity,
            alpha=self.config.train.per_alpha,
            beta_start=self.config.train.per_beta_start,
            beta_frames=self.config.train.per_beta_frames,
            eps=self.config.train.per_eps
        )

        # Learner worker
        self.learner = ZumaLearnerWorker(
            config=self.config,
            replay_buffer=self.replay_buffer,
            stop_event=self.stop_event
        )

        # Actor worker
        self.actor = ZumaActorWorker(
            config=self.config,
            replay_buffer=self.replay_buffer,
            stop_event=self.stop_event,
            get_latest_weights_fn=self.learner.get_latest_weights,
            get_epsilon_fn=self.learner.get_current_epsilon
        )

    def start(self):
        """
        Start the concurrent training pipeline.
        """
        print("=" * 60)
        print("Starting PopCap Zuma 1.0 End-to-End RL Training System")
        print(f"Observation: RGB {self.config.env.obs_width}x{self.config.env.obs_height} x {self.config.env.num_stack_frames} frames ({self.config.env.obs_channels} channels)")
        print(f"Action Space: {self.config.env.total_actions} actions (32x24 spatial grid + Right Click + No-op)")
        print(f"Decision Frequency: {self.config.env.decision_hz} Hz")
        print(f"Learner Device: {self.config.train.device}")
        print("Safety: Press F12 or Ctrl+C at any time for Emergency Stop.")
        print("=" * 60)

        # Start workers
        self.learner.start()
        self.actor.start()

        def signal_handler(sig, frame):
            print("\n[trainer] Shutdown signal received, terminating workers...")
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)

        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[trainer] KeyboardInterrupt caught.")
            self.stop_event.set()
        finally:
            print("[trainer] Waiting for Actor and Learner workers to finish...")
            self.actor.join(timeout=5.0)
            self.learner.join(timeout=5.0)
            print("[trainer] Training system shutdown complete.")
