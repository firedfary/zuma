import unittest
import os
import time
import threading
import torch
import numpy as np

from zuma_rl.config import ZumaConfig, WindowConfig, EnvConfig, ModelConfig, TrainConfig
from zuma_rl.model.spatial_fcn_dqn import SpatialFCNDQN
from zuma_rl.buffer.prioritized_replay import PrioritizedReplayBuffer, SumTree
from zuma_rl.agent.dqn_agent import ZumaDQNAgent
from zuma_rl.env.digit_reader import ZumaDigitReader
from zuma_rl.env.state_machine import ZumaStateMachine, GameState
from zuma_rl.env.zuma_gym_env import ZumaGymEnv
from zuma_rl.window.capture import ZumaScreenCapture, ensure_default_desktop, find_zuma_window
from zuma_rl.window.input_controller import ZumaInputController

class TestZumaRLSuite(unittest.TestCase):
    def setUp(self):
        self.config = ZumaConfig()

    def test_01_config_validation(self):
        """Validate config values match technical specifications"""
        cfg = self.config
        self.assertEqual(cfg.env.obs_width, 240)
        self.assertEqual(cfg.env.obs_height, 180)
        self.assertEqual(cfg.env.num_stack_frames, 4)
        self.assertEqual(cfg.env.obs_channels, 12)
        self.assertEqual(cfg.env.grid_cols, 32)
        self.assertEqual(cfg.env.grid_rows, 24)
        self.assertEqual(cfg.env.num_spatial_actions, 768)
        self.assertEqual(cfg.env.num_aux_actions, 2)
        self.assertEqual(cfg.env.total_actions, 770)
        self.assertEqual(cfg.env.decision_hz, 10.0)
        self.assertEqual(cfg.env.death_penalty, -100.0)
        self.assertEqual(cfg.env.action_bounds, (20, 55, 620, 460))
        self.assertEqual(cfg.env.playable_area, (20, 55, 620, 460))
        self.assertIsNone(cfg.window.action_bounds)
        self.assertIsNone(cfg.window.playable_area)

        # Test playable_area setter
        cfg.env.playable_area = (25, 60, 615, 455)
        self.assertEqual(cfg.env.action_bounds, (25, 60, 615, 455))
        cfg.window.playable_area = (30, 50, 610, 450)
        self.assertEqual(cfg.window.action_bounds, (30, 50, 610, 450))

        # Test invalid action_bounds raises ValueError
        with self.assertRaises(ValueError):
            EnvConfig(action_bounds=(600, 450, 30, 60))
        with self.assertRaises(ValueError):
            WindowConfig(action_bounds=(10, 20, 30))

    def test_02_model_architecture(self):
        """Test SpatialFCNDQN forward pass, tensor shapes, and gradient flow"""
        model = SpatialFCNDQN(self.config.model)
        batch_size = 4
        x = torch.randn(batch_size, 12, 180, 240)
        q_vals, spatial_q, aux_q = model(x)

        # Check output shapes
        self.assertEqual(q_vals.shape, (batch_size, 770))
        self.assertEqual(spatial_q.shape, (batch_size, 24, 32))
        self.assertEqual(aux_q.shape, (batch_size, 2))

        # Test backward pass
        loss = q_vals.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertFalse(torch.isnan(p.grad).any())

    def test_03_prioritized_replay_buffer(self):
        """Test PER SumTree, push, sample, and priority update"""
        buf = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta_start=0.4)
        
        # Push 100 mock transitions
        for i in range(100):
            s = np.zeros((12, 180, 240), dtype=np.float32) + i * 0.001
            a = i % 770
            r = float(i)
            s_next = s + 0.01
            done = (i % 20 == 0)
            buf.push(s, a, r, s_next, done)

        self.assertEqual(len(buf), 100)

        # Sample batch of 16
        batch_size = 16
        states, actions, rewards, next_states, dones, indices, weights = buf.sample(batch_size)
        self.assertEqual(states.shape, (batch_size, 12, 180, 240))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, 12, 180, 240))
        self.assertEqual(dones.shape, (batch_size,))
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(weights.shape, (batch_size,))

        # Update priorities
        new_priorities = np.random.uniform(0.1, 10.0, size=batch_size)
        buf.update_priorities(indices, new_priorities)
        self.assertGreater(buf.tree.total(), 0.0)

    def test_04_agent_and_checkpoint(self):
        """Test Double DQN agent update step, action selection, and checkpoint saving/loading"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = ZumaDQNAgent(self.config.model, self.config.train, device=device)
        
        # Test action selection
        obs = np.random.uniform(0, 1, size=(12, 180, 240)).astype(np.float32)
        act = agent.select_action(obs, evaluate=True)
        self.assertTrue(0 <= act < 770)

        # Test update step with dummy batch
        batch_size = 8
        states = torch.randn(batch_size, 12, 180, 240)
        actions = torch.randint(0, 770, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 12, 180, 240)
        dones = torch.zeros(batch_size)
        indices = list(range(batch_size))
        weights = torch.ones(batch_size)
        batch = (states, actions, rewards, next_states, dones, indices, weights)

        loss, td_errors = agent.update_step(batch)
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        self.assertEqual(len(td_errors), batch_size)

        # Test checkpoint saving and loading
        ckpt_path = r"d:\zuma\test_ckpt.pt"
        agent.save_checkpoint(ckpt_path, {"test_meta": 12345})
        self.assertTrue(os.path.exists(ckpt_path))

        new_agent = ZumaDQNAgent(self.config.model, self.config.train, device=device)
        new_agent.load_checkpoint(ckpt_path)
        self.assertEqual(new_agent.train_step_count, agent.train_step_count)
        
        # Verify parameter match
        for p1, p2 in zip(agent.q_net.parameters(), new_agent.q_net.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_05_digit_reader(self):
        """Test ZumaDigitReader template loading, proper NMS, and multi-digit score extraction"""
        reader = ZumaDigitReader()
        self.assertGreater(len(reader.templates), 0)
        
        # Test synthetic multi-digit score extraction (e.g. 54320 and 10896)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        x_cursor = 300
        for d in [5, 4, 3, 2, 0]:
            tmpl = reader.templates[d]
            th, tw = tmpl.shape
            for r in range(th):
                for c in range(tw):
                    if tmpl[r, c] > 100:
                        canvas[4 + r, x_cursor + c] = [255, 230, 0]
            x_cursor += tw + 2

        score = reader.extract_score(canvas, roi=(2, 22, 265, 420))
        self.assertEqual(score, 54320, f"Expected 54320, got {score}")

        # Test another number
        canvas2 = np.zeros((480, 640, 3), dtype=np.uint8)
        x_cursor = 310
        for d in [1, 0, 8, 9, 6]:
            tmpl = reader.templates[d]
            th, tw = tmpl.shape
            for r in range(th):
                for c in range(tw):
                    if tmpl[r, c] > 100:
                        canvas2[4 + r, x_cursor + c] = [255, 230, 0]
            x_cursor += tw + 2

        score2 = reader.extract_score(canvas2, roi=(2, 22, 265, 420))
        self.assertEqual(score2, 10896, f"Expected 10896, got {score2}")

    def test_06_state_machine_detection(self):
        """Test state machine discrimination across SETTLEMENT, SURVIVAL_MENU, PAUSED, and IN_GAME"""
        sm = ZumaStateMachine(game_dir=self.config.game_dir)
        self.assertGreater(len(sm.templates), 0)

        import cv2
        # 1. Test settlement frame
        if os.path.exists(r"d:\zuma\test_client_only.png"):
            img = cv2.imdecode(np.fromfile(r"d:\zuma\test_client_only.png", dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            state, info = sm.detect_state(rgb)
            self.assertEqual(state, GameState.SETTLEMENT)
            self.assertIsNotNone(info["dialog_btn_loc"])

        # 2. Test survival menu frame
        if os.path.exists(r"d:\zuma\survival_menu_client_only.png"):
            img = cv2.imdecode(np.fromfile(r"d:\zuma\survival_menu_client_only.png", dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            state, info = sm.detect_state(rgb)
            self.assertEqual(state, GameState.SURVIVAL_MENU)
            self.assertIsNotNone(info["play_btn_loc"])

        # 3. Test paused frame
        if os.path.exists(r"d:\zuma\current_game_screen.png"):
            img = cv2.imdecode(np.fromfile(r"d:\zuma\current_game_screen.png", dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            state, info = sm.detect_state(rgb)
            self.assertEqual(state, GameState.PAUSED)

        # 4. Test in-game active frame
        if os.path.exists(r"d:\zuma\test_balls_rolling.png"):
            img = cv2.imdecode(np.fromfile(r"d:\zuma\test_balls_rolling.png", dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            state, info = sm.detect_state(rgb)
            self.assertEqual(state, GameState.IN_GAME)

    def test_07_action_mapping(self):
        """Test coordinate mapping of all 770 actions within safe playable bounds"""
        safe_win_cfg = WindowConfig(auto_activate_window=False)
        controller = ZumaInputController(window_config=safe_win_cfg, env_config=self.config.env)
        min_x, min_y, max_x, max_y = self.config.env.action_bounds

        # 1. Test corner coordinates execution (with dry_run=True to prevent desktop clicks)
        for act in [0, 31, 767, 768, 769]:
            act_type, coords = controller.execute_action(act, dry_run=True)
            if act < 768:
                self.assertEqual(act_type, "click")
                x, y = coords
                self.assertTrue(0 <= x <= 640)
                self.assertTrue(0 <= y <= 480)
                # Ensure coordinates strictly fall inside safe playable bounds
                self.assertTrue(min_x <= x <= max_x, f"Action {act} x={x} outside [{min_x}, {max_x}]")
                self.assertTrue(min_y <= y <= max_y, f"Action {act} y={y} outside [{min_y}, {max_y}]")
                # Ensure top row coordinates are safely below the top menu bar (y < 50)
                self.assertGreaterEqual(y, 55, f"Action {act} y={y} risks clicking top menu button!")
            elif act == 768:
                self.assertEqual(act_type, "swap")
            else:
                self.assertEqual(act_type, "noop")

        # 2. Exhaustively verify all 768 spatial actions fall within safe bounds without menu overlap
        for act in range(self.config.env.num_spatial_actions):
            act_type, (x, y) = controller.get_action_coords(act)
            self.assertEqual(act_type, "click")
            self.assertTrue(min_x <= x <= max_x)
            self.assertTrue(min_y <= y <= max_y)
            self.assertGreaterEqual(y, 55, f"Spatial action {act} y={y} risks top bar menu click!")

        # 3. Verify custom EnvConfig action_bounds configuration
        custom_env_cfg = EnvConfig(action_bounds=(30, 60, 600, 450))
        custom_controller = ZumaInputController(window_config=safe_win_cfg, env_config=custom_env_cfg)
        _, (c_x0, c_y0) = custom_controller.get_action_coords(0)
        self.assertGreaterEqual(c_x0, 30)
        self.assertGreaterEqual(c_y0, 60)

        # 4. Verify WindowConfig.action_bounds takes precedence over EnvConfig.action_bounds
        override_win_cfg = WindowConfig(auto_activate_window=False, action_bounds=(40, 70, 590, 440))
        override_controller = ZumaInputController(window_config=override_win_cfg, env_config=custom_env_cfg)
        _, (o_x0, o_y0) = override_controller.get_action_coords(0)
        self.assertGreaterEqual(o_x0, 40)
        self.assertGreaterEqual(o_y0, 70)

        # 5. Verify graceful fallback to full client area when action_bounds is None
        none_env_cfg = EnvConfig(action_bounds=None)
        none_win_cfg = WindowConfig(auto_activate_window=False, action_bounds=None)
        none_controller = ZumaInputController(window_config=none_win_cfg, env_config=none_env_cfg)
        _, (n_x0, n_y0) = none_controller.get_action_coords(0)
        self.assertEqual(n_x0, 10)  # (0 + 0.5) * (640 / 32) = 10
        self.assertEqual(n_y0, 10)  # (0 + 0.5) * (480 / 24) = 10

        # 6. Verify invalid action indices raise ValueError
        with self.assertRaises(ValueError):
            controller.get_action_coords(-1)
        with self.assertRaises(ValueError):
            controller.get_action_coords(770)
        with self.assertRaises(ValueError):
            controller.get_action_coords(1000)
        with self.assertRaises(ValueError):
            controller.execute_action(-1, dry_run=True)

    def test_08_multithreaded_concurrency(self):
        """Test thread safety of PrioritizedReplayBuffer with concurrent writers and readers"""
        buf = PrioritizedReplayBuffer(capacity=2000)
        stop_flag = threading.Event()

        def writer():
            for i in range(200):
                if stop_flag.is_set():
                    break
                s = np.zeros((12, 180, 240), dtype=np.float32)
                buf.push(s, i % 770, float(i), s, False)
                time.sleep(0.001)

        def reader():
            while not stop_flag.is_set():
                if len(buf) >= 16:
                    batch = buf.sample(16)
                    buf.update_priorities(batch[5], np.ones(16))
                time.sleep(0.002)

        t_writer = threading.Thread(target=writer)
        t_reader = threading.Thread(target=reader)
        t_writer.start()
        t_reader.start()

        t_writer.join(timeout=3.0)
        stop_flag.set()
        t_reader.join(timeout=3.0)

        self.assertGreater(len(buf), 50)

    def test_09_buffer_safety_and_edge_cases(self):
        """Test PER buffer guards against undersized sampling and NaN priorities"""
        buf = PrioritizedReplayBuffer(capacity=500)
        # 1. Test sampling with empty buffer raises ValueError
        with self.assertRaises(ValueError):
            buf.sample(16)

        # 2. Add 5 items, try sampling 16 (should raise ValueError)
        for i in range(5):
            s = np.zeros((12, 180, 240), dtype=np.float32)
            buf.push(s, i, 1.0, s, False)
        with self.assertRaises(ValueError):
            buf.sample(16)

        # 3. Add up to 32 items and sample 16
        for i in range(5, 32):
            s = np.zeros((12, 180, 240), dtype=np.float32)
            buf.push(s, i, 1.0, s, False)
        batch = buf.sample(16)
        self.assertEqual(len(batch[5]), 16)

        # 4. Update priorities with NaN and Inf to verify sanitization
        bad_priorities = np.array([np.nan, np.inf, -np.inf, 2.5] * 4)
        buf.update_priorities(batch[5], bad_priorities)
        self.assertFalse(np.isnan(buf.tree.total()))
        self.assertFalse(np.isinf(buf.tree.total()))
        self.assertGreater(buf.tree.total(), 0.0)

    def test_10_client_capture_dimensions(self):
        """Test ZumaScreenCapture returns exact 640x480 client image without title bar offset"""
        cap = ZumaScreenCapture()
        if cap.is_window_valid():
            frame = cap.capture_frame()
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertGreater(frame.mean(), 1.0)

if __name__ == "__main__":
    unittest.main()
