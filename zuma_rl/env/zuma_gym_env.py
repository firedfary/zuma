import time
from collections import deque
from typing import Optional, Tuple, Dict, Any
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..config import ZumaConfig, EnvConfig, WindowConfig
from ..window.capture import ZumaScreenCapture
from ..window.input_controller import ZumaInputController
from .digit_reader import ZumaDigitReader
from .state_machine import ZumaStateMachine, GameState

class ZumaGymEnv(gym.Env):
    """
    Gymnasium environment for PopCap Zuma 1.0.
    - Observation: (12, 180, 240) float32 in [0, 1] (4 RGB frames stacked)
    - Action: Discrete(770) (32x24 spatial grid clicks + Right Click Swap + No-op)
    - Decision Rate: 10 Hz (100 ms per step)
    - Reward: Composite shaped reward (ΔScore, empty shot penalty, death penalty)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: Optional[ZumaConfig] = None):
        super().__init__()
        self.config = config or ZumaConfig()
        self.e_config = self.config.env
        self.w_config = self.config.window

        # Components
        self.capture = ZumaScreenCapture(
            expected_w=self.w_config.expected_client_width,
            expected_h=self.w_config.expected_client_height
        )
        self.controller = ZumaInputController(
            hwnd=self.capture.hwnd,
            window_config=self.w_config,
            env_config=self.e_config
        )
        self.digit_reader = ZumaDigitReader()
        self.state_machine = ZumaStateMachine(
            game_dir=self.config.game_dir,
            capture=self.capture,
            controller=self.controller
        )

        # Gymnasium spaces
        self.action_space = spaces.Discrete(self.e_config.total_actions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.e_config.obs_channels, self.e_config.obs_height, self.e_config.obs_width),
            dtype=np.float32
        )

        # Frame stacking deque: holds 4 frames of shape (3, 180, 240)
        self.frame_buffer = deque(maxlen=self.e_config.num_stack_frames)

        # Episode tracking variables
        self.current_score = 0
        self.prev_score = 0
        self.episode_steps = 0
        self.episode_score_gain = 0
        self.last_step_time = time.time()

    def _process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Downsample 640x480 frame to (240, 180) and reorder to (3, 180, 240) normalized float32.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return np.zeros((3, self.e_config.obs_height, self.e_config.obs_width), dtype=np.float32)

        # Resize to (width=240, height=180)
        resized = cv2.resize(
            frame_rgb,
            (self.e_config.obs_width, self.e_config.obs_height),
            interpolation=cv2.INTER_AREA
        )
        # Convert HWC to CHW and normalize to [0, 1]
        chw = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        return chw

    def _get_stacked_obs(self) -> np.ndarray:
        """
        Concatenate the 4 stacked frames into shape (12, 180, 240).
        """
        while len(self.frame_buffer) < self.e_config.num_stack_frames:
            # Fill with blank or earliest frame
            first = self.frame_buffer[0] if len(self.frame_buffer) > 0 else np.zeros(
                (3, self.e_config.obs_height, self.e_config.obs_width), dtype=np.float32
            )
            self.frame_buffer.append(first)
        return np.concatenate(list(self.frame_buffer), axis=0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.episode_steps = 0
        self.episode_score_gain = 0
        self.frame_buffer.clear()

        # Check state machine and auto-reset if needed
        frame = self.capture.capture_frame()
        state, _ = self.state_machine.detect_state(frame)
        if state != GameState.IN_GAME:
            print(f"[env] State is {state.name}, executing auto-reset...")
            self.state_machine.handle_reset_step(max_wait_sec=10.0)

        # Initial frame capture and score reading
        time.sleep(0.1)
        frame = self.capture.capture_frame()
        if frame is None:
            frame = np.zeros((self.w_config.expected_client_height, self.w_config.expected_client_width, 3), dtype=np.uint8)

        self.current_score = self.digit_reader.extract_score(frame, roi=self.e_config.score_roi)
        self.prev_score = self.current_score

        processed = self._process_frame(frame)
        for _ in range(self.e_config.num_stack_frames):
            self.frame_buffer.append(processed)

        obs = self._get_stacked_obs()
        self.last_step_time = time.time()

        info = {
            "initial_score": self.current_score,
            "state": self.state_machine.current_state.name
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step at 10 Hz rate:
        1. Action execution
        2. Wait to satisfy 100ms step duration
        3. Capture new frame and parse score
        4. State detection (check game over)
        5. Composite reward shaping
        """
        self.episode_steps += 1

        # Check emergency stop
        if self.controller.is_emergency_stopped():
            obs = self._get_stacked_obs()
            return obs, 0.0, True, True, {"emergency_stopped": True}

        # 1. Execute action
        act_type, act_coords = self.controller.execute_action(action)

        # 2. Precise 10 Hz timing pacing
        target_step_time = self.last_step_time + self.e_config.step_duration_sec
        now = time.time()
        sleep_dur = target_step_time - now
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        self.last_step_time = time.time()

        # 3. Capture new frame
        raw_frame = self.capture.capture_frame()
        if raw_frame is None:
            raw_frame = np.zeros((self.w_config.expected_client_height, self.w_config.expected_client_width, 3), dtype=np.uint8)

        # Update frame buffer
        processed = self._process_frame(raw_frame)
        self.frame_buffer.append(processed)
        obs = self._get_stacked_obs()

        # 4. Extract score
        new_score = self.digit_reader.extract_score(raw_frame, roi=self.e_config.score_roi)
        delta_score = max(0, new_score - self.current_score)
        self.prev_score = self.current_score
        self.current_score = new_score
        self.episode_score_gain += delta_score

        # 5. State detection (Game Over check)
        state, state_info = self.state_machine.detect_state(raw_frame)
        if state == GameState.PAUSED:
            # Auto unpause if paused
            self.controller.send_input_click(320, 240, delay_sec=0.05, activate=True)
            time.sleep(0.3)
            new_f = self.capture.capture_frame()
            if new_f is not None:
                raw_frame = new_f
            state, state_info = self.state_machine.detect_state(raw_frame)

        is_game_over = (state == GameState.GAME_OVER) or (state == GameState.SETTLEMENT) or (state == GameState.SURVIVAL_MENU)

        # 6. Composite reward shaping
        reward = 0.0
        
        # Base score increment reward
        if delta_score > 0:
            reward += delta_score * self.e_config.delta_score_scale
        
        # Step penalty
        reward += self.e_config.step_penalty

        # Empty shot penalty
        if act_type == "click" and delta_score == 0:
            reward += self.e_config.empty_shot_penalty

        # Survival reward
        if not is_game_over:
            reward += self.e_config.survival_reward

        # Death penalty
        if is_game_over:
            reward += self.e_config.death_penalty

        terminated = is_game_over
        truncated = self.episode_steps >= self.e_config.max_episode_steps

        info = {
            "score": self.current_score,
            "delta_score": delta_score,
            "episode_score_gain": self.episode_score_gain,
            "episode_steps": self.episode_steps,
            "action_type": act_type,
            "action_coords": act_coords,
            "state": state.name
        }

        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.frame_buffer.clear()
