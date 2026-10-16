import unittest
import os
import time
import threading
import torch
import numpy as np
import cv2

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

    def test_11_hole_openness_progression(self):
        """Test skull mouth openness progression is strictly monotonic: Fig 1 <= 0.10, Fig 5 >= 0.85"""
        sm = ZumaStateMachine(
            game_dir=self.config.game_dir,
            danger_config=self.config.danger,
            start_gate_config=self.config.start_gate
        )
        user_p = r"C:/Users/fired/.gemini/antigravity/brain/fa5a9161-23ff-46ba-9697-6cec10ef368c/.user_uploaded"
        fig_files = [
            "media_1792076465643_559df118.jpg",  # Fig 1: closed
            "media_1792076480168_aacca3ee.jpg",  # Fig 2: opening
            "media_1792076492170_757bb65d.jpg",  # Fig 3: opening further
            "media_1792076500374_8083b2ca.jpg",  # Fig 4: opening more
            "media_1792076555026_364c2a5f.jpg",  # Fig 5: fully open
        ]
        if os.path.exists(user_p) and all(os.path.exists(os.path.join(user_p, f)) for f in fig_files):
            openness_vals = []
            for fn in fig_files:
                raw = np.fromfile(os.path.join(user_p, fn), dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                f640 = cv2.resize(img[39:39+720, 2:2+960], (640, 480))
                roi = f640[216-44:216+44, 211-44:211+44]
                val = sm.compute_skull_openness(roi)
                openness_vals.append(val)

            # Assert Fig 1 is <= 0.10
            self.assertLessEqual(openness_vals[0], 0.10, f"Fig 1 expected <= 0.10, got {openness_vals[0]}")
            # Assert Fig 5 is >= 0.85
            self.assertGreaterEqual(openness_vals[-1], 0.85, f"Fig 5 expected >= 0.85, got {openness_vals[-1]}")
            # Assert strictly monotonic progression
            for i in range(len(openness_vals) - 1):
                self.assertLess(
                    openness_vals[i],
                    openness_vals[i+1],
                    f"Progression not monotonic: Fig {i+1} ({openness_vals[i]}) >= Fig {i+2} ({openness_vals[i+1]})"
                )

        # Synthetic cavity verification to ensure unit test coverage in any environment
        synthetic_ratios = [0.0, 0.10, 0.25, 0.50, 0.95]
        synthetic_vals = []
        for r in synthetic_ratios:
            cavity = np.ones((40, 40, 3), dtype=np.uint8) * 100
            dark_pixels = int(40 * 40 * r)
            cavity.reshape(-1, 3)[:dark_pixels] = 10
            val = sm.compute_skull_openness(cavity)
            synthetic_vals.append(val)

        self.assertLessEqual(synthetic_vals[0], 0.05)
        self.assertGreaterEqual(synthetic_vals[-1], 0.85)
        for i in range(len(synthetic_vals) - 1):
            self.assertLess(synthetic_vals[i], synthetic_vals[i+1])

    def test_12_lives_banner_detection(self):
        """Test lives banner ('剩余 X 次机会') is accurately detected without false positives across all gameplay frames"""
        sm = ZumaStateMachine(
            game_dir=self.config.game_dir,
            danger_config=self.config.danger,
            start_gate_config=self.config.start_gate
        )
        user_p = r"C:/Users/fired/.gemini/antigravity/brain/fa5a9161-23ff-46ba-9697-6cec10ef368c/.user_uploaded"
        banner_file = os.path.join(user_p, "media_1792075117894_a12e8f12.jpg")
        gameplay_files = [
            "media_1792074831582_7ba1c893.jpg",  # Problem 1: in hole
            "media_1792076465643_559df118.jpg",  # Fig 1
            "media_1792076480168_aacca3ee.jpg",  # Fig 2
            "media_1792076492170_757bb65d.jpg",  # Fig 3
            "media_1792076500374_8083b2ca.jpg",  # Fig 4
            "media_1792076555026_364c2a5f.jpg",  # Fig 5
        ]

        if os.path.exists(banner_file):
            raw_b = np.fromfile(banner_file, dtype=np.uint8)
            img_b = cv2.imdecode(raw_b, cv2.IMREAD_COLOR)
            f640_b = cv2.resize(img_b[39:39+720, 2:2+960], (640, 480))
            self.assertTrue(sm.check_lives_banner(f640_b), "Failed to detect lives banner in Problem 2 image")

        for gf in gameplay_files:
            fp = os.path.join(user_p, gf)
            if os.path.exists(fp):
                raw = np.fromfile(fp, dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                f640 = cv2.resize(img[39:39+720, 2:2+960], (640, 480))
                self.assertFalse(sm.check_lives_banner(f640), f"False positive lives banner on gameplay frame {gf}")

        # Synthetic banner and clear screen
        canvas_banner = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas_banner[180:240, 200:440] = [30, 200, 255]
        self.assertTrue(sm.check_lives_banner(canvas_banner))

        canvas_empty = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertFalse(sm.check_lives_banner(canvas_empty))

    def test_13_motion_settling_gating(self):
        """Test 4-step composite gating: banner clear -> motion rush -> calm settled -> ready"""
        sm = ZumaStateMachine(
            game_dir=self.config.game_dir,
            danger_config=self.config.danger,
            start_gate_config=self.config.start_gate
        )

        class MockCapture:
            def __init__(self):
                self.call_idx = 0
                self.frames = []
                # Frame 0-1: Banner present
                f_banner = np.zeros((480, 640, 3), dtype=np.uint8)
                f_banner[180:240, 200:440] = [30, 200, 255]
                self.frames.extend([f_banner, f_banner])
                
                # Frame 2-3: Banner gone, high difference (rush motion)
                f_rush1 = np.zeros((480, 640, 3), dtype=np.uint8)
                f_rush2 = np.ones((480, 640, 3), dtype=np.uint8) * 30
                self.frames.extend([f_rush1, f_rush2])

                # Frame 4-7: Settled baseline (diff = 0 < 3.0) for >= 3 frames
                f_calm = np.ones((480, 640, 3), dtype=np.uint8) * 30
                # Frog mouth has ball
                f_calm[233-10:233+10, 327-10:327+10] = [0, 220, 255]
                self.frames.extend([f_calm, f_calm, f_calm, f_calm])

            def capture_frame(self):
                if self.call_idx < len(self.frames):
                    frame = self.frames[self.call_idx]
                    self.call_idx += 1
                    return frame
                return self.frames[-1]

        mock_cap = MockCapture()
        ready = sm.wait_for_round_ready(capture=mock_cap, max_wait_sec=2.0)
        self.assertTrue(ready)
        self.assertEqual(sm.current_state, GameState.IN_GAME)

    def test_14_multi_hole_anchoring(self):
        """Test multi-hole template matching anchoring with NMS for dual-track and single-track maps"""
        sm = ZumaStateMachine(
            game_dir=self.config.game_dir,
            danger_config=self.config.danger,
            start_gate_config=self.config.start_gate
        )
        self.assertIn("hole_template", sm.templates)
        self.assertIn("hole_mask", sm.templates)

        tmpl = sm.templates["hole_template"]
        mask = sm.templates["hole_mask"]
        th_h, th_w = tmpl.shape[:2]

        canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 128
        # Place two hole locations (dual-track level)
        loc1 = (120, 160)
        loc2 = (420, 280)

        for loc in [loc1, loc2]:
            x, y = loc
            for r in range(th_h):
                for c in range(th_w):
                    if mask[r, c] > 50:
                        canvas[y + r, x + c] = tmpl[r, c]

        anchored = sm.anchor_hole_locations(canvas, threshold=0.65)
        self.assertEqual(len(anchored), 2, f"Expected 2 holes for dual track, found {len(anchored)}")
        expected_centers = [(loc1[0] + th_w // 2, loc1[1] + th_h // 2), (loc2[0] + th_w // 2, loc2[1] + th_h // 2)]
        
        for exp in expected_centers:
            matched = any(np.hypot(cx - exp[0], cy - exp[1]) <= 5 for cx, cy in anchored)
            self.assertTrue(matched, f"Expected hole at {exp} not found in {anchored}")

        # Real user screenshot test: verify single-track map anchors EXACTLY 1 hole without false second hole
        user_p = r"C:/Users/fired/.gemini/antigravity/brain/fa5a9161-23ff-46ba-9697-6cec10ef368c/.user_uploaded"
        p_real = os.path.join(user_p, "media_1792074831582_7ba1c893.jpg")
        if os.path.exists(p_real):
            raw = np.fromfile(p_real, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            f640 = cv2.resize(img[39:39+720, 2:2+960], (640, 480))
            rgb = cv2.cvtColor(f640, cv2.COLOR_BGR2RGB)
            real_anchored = sm.anchor_hole_locations(rgb)
            self.assertEqual(len(real_anchored), 1, f"Expected exactly 1 hole on single track level, got {real_anchored}")
            self.assertTrue(abs(real_anchored[0][0] - 211) <= 3 and abs(real_anchored[0][1] - 216) <= 3)

    def test_15_dying_hole_and_hysteresis(self):
        """Test hysteresis recovery (openness < 0.70) and fatal termination (DYING_HOLE) on real and synthetic images"""
        sm = ZumaStateMachine(
            game_dir=self.config.game_dir,
            danger_config=self.config.danger,
            start_gate_config=self.config.start_gate
        )
        sm.hole_locations = [(211, 216)]

        # 1. Test real Problem 1 (ball entered hole) triggers DYING_HOLE with max_openness = 1.0
        user_p = r"C:/Users/fired/.gemini/antigravity/brain/fa5a9161-23ff-46ba-9697-6cec10ef368c/.user_uploaded"
        p_fatal = os.path.join(user_p, "media_1792074831582_7ba1c893.jpg")
        if os.path.exists(p_fatal):
            raw = np.fromfile(p_fatal, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            f640 = cv2.resize(img[39:39+720, 2:2+960], (640, 480))
            rgb = cv2.cvtColor(f640, cv2.COLOR_BGR2RGB)
            sm.is_emergency = False
            state, info = sm.detect_state(rgb)
            self.assertEqual(state, GameState.DYING_HOLE, "Problem 1 must trigger DYING_HOLE")
            self.assertEqual(info["max_openness"], 1.0)
            self.assertTrue(info["is_emergency"])

        # 2. Test real Fig 5 (skull fully open, ball not in hole) triggers EMERGENCY
        p_open = os.path.join(user_p, "media_1792076555026_364c2a5f.jpg")
        if os.path.exists(p_open):
            raw = np.fromfile(p_open, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            f640 = cv2.resize(img[39:39+720, 2:2+960], (640, 480))
            rgb = cv2.cvtColor(f640, cv2.COLOR_BGR2RGB)
            sm.is_emergency = False
            state, info = sm.detect_state(rgb)
            self.assertEqual(state, GameState.EMERGENCY, "Fig 5 must trigger EMERGENCY")
            self.assertGreaterEqual(info["max_openness"], 0.85)

        # 3. Test real Fig 1 (closed skull) is IN_GAME
        p_closed = os.path.join(user_p, "media_1792076465643_559df118.jpg")
        if os.path.exists(p_closed):
            raw = np.fromfile(p_closed, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            f640 = cv2.resize(img[39:39+720, 2:2+960], (640, 480))
            rgb = cv2.cvtColor(f640, cv2.COLOR_BGR2RGB)
            sm.is_emergency = False
            state, info = sm.detect_state(rgb)
            self.assertEqual(state, GameState.IN_GAME, "Fig 1 must be IN_GAME")
            self.assertLessEqual(info["max_openness"], 0.10)

        # 4. Multi-color ball in hole detection (Blue, Green, Red, Purple)
        for ball_bgr in [[255, 50, 50], [50, 255, 50], [50, 50, 255], [200, 50, 200]]:
            cavity_roi = np.zeros((88, 88, 3), dtype=np.uint8)
            cavity_roi[44-12:44+12, 44-12:44+12] = ball_bgr
            self.assertTrue(sm.check_ball_in_hole(cavity_roi), f"Ball color {ball_bgr} should be detected in hole")

        # 5. Test hysteresis hold when openness drops to 0.75 (above 0.70 threshold)
        sm.hole_locations = [(200, 200)]
        sm.is_emergency = True
        frame_hold = np.zeros((480, 640, 3), dtype=np.uint8) + 100
        frame_hold[200-20:200+20, 200-20:200+20] = 0
        frame_hold[200-8:200+8, 200-8:200+8] = 80
        frame_hold[233-10:233+10, 327-10:327+10] = [0, 220, 255]

        state, info = sm.detect_state(frame_hold)
        self.assertEqual(state, GameState.EMERGENCY, "Hysteresis should keep EMERGENCY while >= 0.70")
        self.assertTrue(sm.is_emergency)

        # 6. Test hysteresis recovery when openness drops < 0.70
        frame_recovery = np.zeros((480, 640, 3), dtype=np.uint8) + 100
        frame_recovery[200-20:200+20, 200-20:200+20] = 120
        frame_recovery[233-10:233+10, 327-10:327+10] = [0, 220, 255]

        state, info = sm.detect_state(frame_recovery)
        self.assertEqual(state, GameState.IN_GAME, "Hysteresis should recover to IN_GAME when < 0.70")
        self.assertFalse(sm.is_emergency)

    def test_16_gym_env_danger_gating_and_termination(self):
        """Test ZumaGymEnv integration of danger reward shaping and fatal DYING_HOLE input blocking"""
        cfg = ZumaConfig()
        cfg.window.auto_activate_window = False
        env = ZumaGymEnv(config=cfg)

        # 1. Verify reset provides danger info
        obs, info = env.reset()
        self.assertIn("openness", info)
        self.assertIn("danger_str", info)
        self.assertEqual(info["danger_str"], "Danger: 0%")

        # 2. Verify step returns danger_str formatting (e.g. Danger: 78%)
        env.state_machine.detect_state = lambda f: (GameState.IN_GAME, {"max_openness": 0.78})
        # Execute no-op action (769)
        obs, reward, terminated, truncated, step_info = env.step(769)
        self.assertEqual(step_info["danger_str"], "Danger: 78%")
        self.assertEqual(step_info["danger_pct"], 78)
        self.assertAlmostEqual(step_info["danger_penalty"], cfg.danger.danger_penalty_weight * 0.78)

        # 3. Verify fatal DYING_HOLE physical block
        env.state_machine.current_state = GameState.DYING_HOLE
        obs, reward, terminated, truncated, fatal_info = env.step(100)
        self.assertTrue(terminated)
        self.assertEqual(fatal_info["action_type"], "blocked")
        self.assertEqual(reward, cfg.env.death_penalty)

        env.close()

    def test_17_detect_all_balls(self):
        """Test full-screen ball detection covering 6 colors and real gameplay images"""
        sm = ZumaStateMachine(danger_config=self.config.danger)

        # 1. Edge cases: empty/None frame
        self.assertEqual(sm.detect_all_balls(None), [])
        self.assertEqual(sm.detect_all_balls(np.zeros((0, 0, 3), dtype=np.uint8)), [])
        self.assertEqual(sm.detect_all_balls(np.zeros((480, 640, 3), dtype=np.uint8)), [])

        # 2. Synthetic frame with all 6 ball colors
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        colors_rgb = [
            (255, 0, 0),     # Red
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (200, 0, 200),   # Purple
            (240, 240, 240), # White
        ]
        expected_locs = []
        for i, col in enumerate(colors_rgb):
            pt = (120 + i * 60, 220)
            expected_locs.append(pt)
            cv2.circle(canvas, pt, 15, col, -1)

        detected = sm.detect_all_balls(canvas, is_bgr=False)
        self.assertEqual(len(detected), 6, f"Expected 6 detected balls, got {len(detected)}: {detected}")
        for exp in expected_locs:
            matched = any(np.hypot(exp[0] - d[0], exp[1] - d[1]) <= 3 for d in detected)
            self.assertTrue(matched, f"Ball at {exp} not found in {detected}")

        # 3. Real user screenshots
        user_p = r"C:/Users/fired/.gemini/antigravity/brain/fa5a9161-23ff-46ba-9697-6cec10ef368c/.user_uploaded"
        p_fig1 = os.path.join(user_p, "media_1792076465643_559df118.jpg")
        if os.path.exists(p_fig1):
            raw = np.fromfile(p_fig1, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            f640 = cv2.resize(img[39:39+720, 2:2+960], (640, 480))
            rgb = cv2.cvtColor(f640, cv2.COLOR_BGR2RGB)
            balls = sm.detect_all_balls(rgb, is_bgr=False)
            self.assertGreaterEqual(len(balls), 40, f"Fig 1 should detect at least 40 balls, found {len(balls)}")

    def test_18_collective_ball_motion_and_local_elimination(self):
        """Test collective ball motion accurately distinguishes local elimination from fatal full-chain suction"""
        sm = ZumaStateMachine(danger_config=self.config.danger)

        # 1. Edge cases
        self.assertEqual(sm.compute_collective_ball_motion(None, None, [], []), 0.0)

        # Setup 10 baseline balls
        f0 = np.zeros((480, 640, 3), dtype=np.uint8) + 60
        balls0 = [(100 + i * 32, 200) for i in range(10)]
        for b in balls0:
            cv2.circle(f0, b, 14, (0, 0, 255), -1)

        # 2. Stationary balls -> 0.0
        r_stat = sm.compute_collective_ball_motion(f0, f0, balls0, balls0)
        self.assertEqual(r_stat, 0.0)

        # 3. Slow forward track progression (1 px shift) -> 0.0
        f_slow = np.zeros((480, 640, 3), dtype=np.uint8) + 60
        balls_slow = [(x + 1, y) for x, y in balls0]
        for b in balls_slow:
            cv2.circle(f_slow, b, 14, (0, 0, 255), -1)
        r_slow = sm.compute_collective_ball_motion(f_slow, f0, balls_slow, balls0)
        self.assertEqual(r_slow, 0.0)

        # 4. Local elimination gap closure: 2 balls move backwards by 8 px, 8 balls stationary
        f_local = np.zeros((480, 640, 3), dtype=np.uint8) + 60
        balls_local = balls0[:8] + [(balls0[8][0] - 8, balls0[8][1]), (balls0[9][0] - 8, balls0[9][1])]
        for b in balls_local:
            cv2.circle(f_local, b, 14, (0, 0, 255), -1)
        r_local = sm.compute_collective_ball_motion(f_local, f0, balls_local, balls0)
        self.assertLess(r_local, self.config.danger.collective_fast_ball_ratio)
        self.assertAlmostEqual(r_local, 0.20, places=2)

        # 5. Full-chain acceleration (all 10 balls moving at high speed >= 10 px) -> 1.0 >= 0.80
        f_fast = np.zeros((480, 640, 3), dtype=np.uint8) + 60
        balls_fast = [(x + 10, y) for x, y in balls0]
        for b in balls_fast:
            cv2.circle(f_fast, b, 14, (0, 0, 255), -1)
        r_fast = sm.compute_collective_ball_motion(f_fast, f0, balls_fast, balls0)
        self.assertGreaterEqual(r_fast, self.config.danger.collective_fast_ball_ratio)
        self.assertEqual(r_fast, 1.0)

    def test_19_openness_rate_limiter_and_safe_gating(self):
        """Test rate limiter suppresses instantaneous 0% -> 100% jumping and safe zone blocks flying bullets"""
        sm = ZumaStateMachine(danger_config=self.config.danger)
        sm.hole_locations = [(211, 216)]

        # 1. Rate limiter clamp: instantaneous target spike of 1.0 from 0.0 is clamped by max_openness_step_delta (0.08)
        sm.smoothed_openness = 0.0
        val1 = sm.update_openness_rate_limit(1.0)
        self.assertAlmostEqual(val1, 0.08, places=4)
        val2 = sm.update_openness_rate_limit(1.0)
        self.assertAlmostEqual(val2, 0.16, places=4)

        # Monotonic multi-step progression takes at least 11 steps to reach EMERGENCY (>= 0.85)
        for _ in range(9):
            sm.update_openness_rate_limit(1.0)
        self.assertGreaterEqual(sm.smoothed_openness, 0.85)

        # 2. Replay user scenario (Step 31): Closed skull (0%), blue bullet flies over hole
        sm.reset()
        sm.hole_locations = [(211, 216)]

        # Frame 0: Baseline closed skull
        frame0 = np.zeros((480, 640, 3), dtype=np.uint8) + 120
        # Closed golden teeth cavity (dark_ratio < 0.05)
        frame0[216-20:216+20, 211-20:211+20] = [30, 180, 220]  # Golden teeth BGR
        # Frog has loaded ball
        frame0[233-10:233+10, 327-10:327+10] = [0, 220, 255]

        state0, info0 = sm.detect_state(frame0)
        self.assertEqual(state0, GameState.IN_GAME)
        self.assertLess(info0["max_openness"], self.config.danger.safe_openness_threshold)

        # Frame 1 (Step 31): Blue shot passing over closed skull
        frame1 = frame0.copy()
        frame1[216-10:216+10, 211-10:211+10] = [255, 0, 0]  # Blue bullet in BGR (RGB blue)

        state1, info1 = sm.detect_state(frame1)
        # MUST remain IN_GAME and must NOT jump to 100% or trigger DYING_HOLE!
        self.assertEqual(state1, GameState.IN_GAME, "Flying bullet over closed skull must NOT trigger DYING_HOLE")
        self.assertFalse(info1["has_ball_in_hole"], "Closed skull must shield has_ball_in_hole")
        self.assertLessEqual(info1["max_openness"], 0.10, "Openness must not jump on closed skull")
        self.assertFalse(sm.is_emergency)

    def test_20_fatal_dying_hole_gating(self):
        """Test refactored DYING_HOLE gating: requires EMERGENCY + collective acceleration (>= 2 frames) or swallowed ball"""
        sm = ZumaStateMachine(danger_config=self.config.danger)
        sm.hole_locations = [(200, 200)]
        sm.is_emergency = True
        sm.smoothed_openness = 0.90

        # Create wide open black cavity
        def make_frame(ball_offset=0, has_frog_ball=True):
            f = np.zeros((480, 640, 3), dtype=np.uint8) + 50
            # Open black cavity (dark_ratio > 0.80)
            f[200-20:200+20, 200-20:200+20] = 5
            # Frog ball
            if has_frog_ball:
                f[233-10:233+10, 327-10:327+10] = [0, 220, 255]
            # Chain of 6 balls
            for i in range(6):
                cv2.circle(f, (300 + i * 32 + ball_offset, 300), 14, (0, 0, 255), -1)
            return f

        # Frame 1: Emergency, normal slow progression (offset 0 -> 1)
        f_init = make_frame(ball_offset=0, has_frog_ball=True)
        state, info = sm.detect_state(f_init)
        self.assertEqual(state, GameState.EMERGENCY)

        f_slow = make_frame(ball_offset=1, has_frog_ball=True)
        state, info = sm.detect_state(f_slow)
        self.assertEqual(state, GameState.EMERGENCY, "Slow movement must remain EMERGENCY")

        # Frame 2: First frame of collective acceleration (offset +10) -> consecutive frames = 1 -> remains EMERGENCY
        f_fast1 = make_frame(ball_offset=11, has_frog_ball=True)
        state, info = sm.detect_state(f_fast1)
        self.assertEqual(state, GameState.EMERGENCY, "1 frame of fast motion must not terminate immediately (needs 2)")
        self.assertEqual(sm.fast_motion_consecutive_frames, 1)

        # Frame 3: Second consecutive frame of collective acceleration (offset +21) -> consecutive frames = 2 -> DYING_HOLE!
        f_fast2 = make_frame(ball_offset=21, has_frog_ball=True)
        state, info = sm.detect_state(f_fast2)
        self.assertEqual(state, GameState.DYING_HOLE, "2 consecutive fast motion frames in EMERGENCY must trigger DYING_HOLE")
        self.assertEqual(info["max_openness"], 1.0)

    def test_21_shooting_in_emergency_does_not_die(self):
        """Test that firing a shot while in EMERGENCY (empty frog mouth) does NOT kill the player when no ball is in hole"""
        sm = ZumaStateMachine(danger_config=self.config.danger)
        sm.hole_locations = [(200, 200)]
        sm.is_emergency = True
        sm.smoothed_openness = 0.90

        # Frame with wide open cavity, balls stationary on track, but frog mouth empty (shot in flight)
        f = np.zeros((480, 640, 3), dtype=np.uint8) + 50
        # Wide open cavity
        f[200-20:200+20, 200-20:200+20] = 5
        # Stationary balls
        for i in range(8):
            cv2.circle(f, (300 + i * 32, 300), 14, (0, 0, 255), -1)
        # Frog mouth at (327, 233) has NO loaded ball (empty background)

        state, info = sm.detect_state(f)
        self.assertEqual(state, GameState.EMERGENCY, "Shooting a ball in emergency must NOT trigger DYING_HOLE!")
        self.assertFalse(info["frog_has_ball"])
        self.assertFalse(info["has_ball_in_hole"])
        self.assertTrue(sm.is_emergency)

    def test_22_rush_motion_at_ball_spacing_shift(self):
        """Test that whole-chain collective rush motion is detected even at shifts matching exact ball diameter / spacing (32px)"""
        sm = ZumaStateMachine(danger_config=self.config.danger)
        f0 = np.zeros((480, 640, 3), dtype=np.uint8) + 60
        balls0 = [(100 + i * 32, 200) for i in range(10)]
        for b in balls0:
            cv2.circle(f0, b, 14, (0, 0, 255), -1)

        # Shift by exact ball spacing of 32 pixels
        f_shift32 = np.zeros((480, 640, 3), dtype=np.uint8) + 60
        balls_shift32 = [(x + 32, y) for x, y in balls0]
        for b in balls_shift32:
            cv2.circle(f_shift32, b, 14, (0, 0, 255), -1)

        ratio32 = sm.compute_collective_ball_motion(f_shift32, f0, balls_shift32, balls0)
        self.assertGreaterEqual(ratio32, self.config.danger.collective_fast_ball_ratio, "32px shift must detect collective fast motion >= 0.80")
        self.assertEqual(ratio32, 1.0)

    def test_23_rate_limiter_active_on_first_step(self):
        """Test that rate limiter is active on step 1 of an episode after reset(), clamping sudden spikes"""
        sm = ZumaStateMachine(danger_config=self.config.danger)
        sm.reset()
        sm.hole_locations = [(211, 216)]

        # Frame with sudden artificial openness spike (1.0) on step 1
        frame_spike = np.zeros((480, 640, 3), dtype=np.uint8) + 50
        frame_spike[216-20:216+20, 211-20:211+20] = 5  # dark cavity = 1.0 openness
        frame_spike[233-10:233+10, 327-10:327+10] = [0, 220, 255]

        state, info = sm.detect_state(frame_spike)
        self.assertAlmostEqual(info["max_openness"], sm.danger_config.max_openness_step_delta, places=2)
        self.assertEqual(state, GameState.IN_GAME, "Step 1 spike must be clamped to safe zone (0.08 < 0.30) and stay IN_GAME")
        self.assertFalse(sm.is_emergency)

if __name__ == "__main__":
    unittest.main()

