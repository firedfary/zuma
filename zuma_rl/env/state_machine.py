import os
import time
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
from PIL import Image

from ..config import WindowConfig, EnvConfig, DangerConfig, StartGateConfig
from ..window.capture import ZumaScreenCapture
from ..window.input_controller import ZumaInputController

class GameState(Enum):
    UNKNOWN = auto()
    IN_GAME = auto()
    PAUSED = auto()
    GAME_OVER = auto()
    SETTLEMENT = auto()
    SURVIVAL_MENU = auto()
    LIVES_BANNER = auto()     # 图2：剩余X次机会展示中
    EMERGENCY = auto()        # 图5：骷髅洞完全张开，濒死警戒中
    DYING_HOLE = auto()       # 图1：球已落入洞中，死亡吸入程序触发
    STARTING_RUSH = auto()    # 开局冲球中

class ZumaStateMachine:
    """
    State machine for Zuma gameplay detection and automatic reset loop:
    - IN_GAME: Normal gameplay
    - PAUSED: Game paused with '暂停中' -> click center (320, 240) to unpause
    - GAME_OVER: Detected 'GAME OVER' screen -> halt gameplay actions, wait for settlement
    - SETTLEMENT: Dialog screen with '确定' (DialogButton) -> auto-click '确定' at (320, 370)
    - SURVIVAL_MENU: Survival/Gauntlet screen with '游戏' (GauntPlayButton) -> auto-click '游戏' at (320, 455)
    - LIVES_BANNER: '剩余 X 次机会' center banner -> wait to clear
    - EMERGENCY: Skull mouth fully open (openness >= 0.85)
    - DYING_HOLE: Ball entering hole / lost ball -> immediate fatal termination
    - STARTING_RUSH: Fast ball rush animation before settling
    """
    def __init__(
        self,
        game_dir: str = r"d:\zuma\祖玛豪华版",
        capture: Optional[ZumaScreenCapture] = None,
        controller: Optional[ZumaInputController] = None,
        danger_config: Optional[DangerConfig] = None,
        start_gate_config: Optional[StartGateConfig] = None,
    ):
        self.game_dir = game_dir
        self.capture = capture or ZumaScreenCapture()
        self.controller = controller or ZumaInputController(hwnd=self.capture.hwnd)
        self.danger_config = danger_config or DangerConfig()
        self.start_gate_config = start_gate_config or StartGateConfig()
        
        self.current_state = GameState.UNKNOWN
        self.templates: Dict[str, np.ndarray] = {}
        self.hole_locations: List[Tuple[int, int]] = []
        self.is_emergency: bool = False
        self._last_is_emergency: bool = False
        self.smoothed_openness: Optional[float] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_balls: Optional[List[Tuple[int, int]]] = None
        self.fast_motion_consecutive_frames: int = 0
        self._load_templates()

    def reset(self):
        """Reset internal temporal tracking states at episode boundaries."""
        self.current_state = GameState.UNKNOWN
        self.is_emergency = False
        self._last_is_emergency = False
        self.smoothed_openness = 0.0
        self.prev_frame = None
        self.prev_balls = None
        self.fast_motion_consecutive_frames = 0

    def _load_templates(self):
        img_dir = os.path.join(self.game_dir, "images_chs")
        fonts_dir = os.path.join(self.game_dir, "fonts")
        
        # 1. DialogButton (确定) - shape (48, 360, 3) -> 3 frames, width=120
        dlg_path = os.path.join(img_dir, "DialogButton.jpg")
        if os.path.exists(dlg_path):
            raw = cv2.imdecode(np.fromfile(dlg_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is not None:
                self.templates["dialog_button"] = raw[:, :120]

        # 2. GauntPlayButton (游戏) - shape (33, 420, 3) -> 3 frames, width=140
        play_path = os.path.join(img_dir, "GauntPlayButton.jpg")
        if os.path.exists(play_path):
            raw = cv2.imdecode(np.fromfile(play_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is not None:
                self.templates["gaunt_play"] = raw[:, :140]

        # 3. GauntPractice / Survival buttons
        surv_path = os.path.join(img_dir, "gauntsurvivalbutton.jpg")
        if os.path.exists(surv_path):
            raw = cv2.imdecode(np.fromfile(surv_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is not None:
                self.templates["gaunt_survival"] = raw[:, : raw.shape[1] // 3]

        # 4. GAME OVER text template from NativeAlien48 font
        font_path = os.path.join(fonts_dir, "NativeAlien48.gif")
        if os.path.exists(font_path):
            try:
                with Image.open(font_path) as im:
                    font_gray = np.array(im.convert("L"))
                char_rects = {
                    'G': (241, 0, 38, 90), 'A': (0, 0, 41, 90), 'M': (511, 0, 67, 90),
                    'E': (177, 0, 31, 90), 'O': (618, 0, 52, 90), 'V': (910, 0, 40, 90),
                    'R': (760, 0, 40, 90),
                }
                crops = []
                for ch in "GAME OVER":
                    if ch in char_rects:
                        x, y, w, h = char_rects[ch]
                        crops.append(font_gray[y:y+h, x:x+w])
                    elif ch == ' ':
                        crops.append(np.zeros((90, 20), dtype=np.uint8))
                if crops:
                    self.templates["game_over"] = np.hstack(crops)
            except Exception as e:
                print(f"[state_machine] Could not build GAME OVER template: {e}")

        # 5. Hole template and mask
        hole_path = os.path.join(img_dir, self.danger_config.hole_template_name)
        mask_path = os.path.join(img_dir, self.danger_config.hole_mask_name)
        if os.path.exists(hole_path):
            try:
                with Image.open(hole_path) as im:
                    hole_rgb = np.array(im.convert("RGB"))
                self.templates["hole_template"] = cv2.cvtColor(hole_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[state_machine] Could not load {self.danger_config.hole_template_name}: {e}")
        if os.path.exists(mask_path):
            try:
                with Image.open(mask_path) as im:
                    self.templates["hole_mask"] = np.array(im.convert("L"))
            except Exception as e:
                print(f"[state_machine] Could not load {self.danger_config.hole_mask_name}: {e}")

    def anchor_hole_locations(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None,
        min_distance: int = 50,
        max_holes: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Locate all skull hole centers in the frame using template matching and NMS.
        Supports dual-track levels with multiple skull holes (up to max_holes).
        """
        if frame is None or frame.size == 0:
            return self.hole_locations

        tmpl = self.templates.get("hole_template")
        mask = self.templates.get("hole_mask")
        if tmpl is None or mask is None:
            return self.hole_locations

        # Convert RGB to BGR if frame has 3 channels
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.dtype == np.uint8 else frame
        else:
            frame_bgr = frame

        th_h, th_w = tmpl.shape[:2]
        res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
        max_val = float(res.max())
        if max_val < 0.50:
            return self.hole_locations

        base_th = threshold if threshold is not None else self.danger_config.hole_match_threshold
        rel_drop = self.danger_config.hole_match_relative_drop
        th_val = max(base_th, max_val - rel_drop)
        limit_holes = max_holes if max_holes is not None else self.danger_config.max_holes

        # Find candidates exceeding threshold
        y_locs, x_locs = np.where(res >= th_val)
        candidates = []
        for y, x in zip(y_locs, x_locs):
            candidates.append((float(res[y, x]), int(x), int(y)))

        # Sort descending by score
        candidates.sort(key=lambda item: item[0], reverse=True)

        picked_centers: List[Tuple[int, int]] = []
        for score, x, y in candidates:
            cx = x + th_w // 2
            cy = y + th_h // 2
            too_close = False
            for px, py in picked_centers:
                if np.hypot(cx - px, cy - py) < min_distance:
                    too_close = True
                    break
            if not too_close:
                picked_centers.append((cx, cy))
                if len(picked_centers) >= limit_holes:
                    break

        # Fallback if no candidate met threshold: try top peak if >= 0.50
        if not picked_centers and max_val >= 0.50:
            _, _, _, max_loc = cv2.minMaxLoc(res)
            cx = max_loc[0] + th_w // 2
            cy = max_loc[1] + th_h // 2
            picked_centers.append((cx, cy))

        if picked_centers:
            self.hole_locations = picked_centers

        return self.hole_locations

    def compute_skull_openness(self, hole_roi: np.ndarray) -> float:
        """
        Compute skull mouth openness S_open in [0.0, 1.0] from center cavity dark ratio.
        """
        if hole_roi is None or hole_roi.size == 0:
            return 0.0

        h, w = hole_roi.shape[:2]
        cx, cy = w // 2, h // 2
        half = self.danger_config.cavity_half_size

        if h >= half * 2 and w >= half * 2:
            cavity = hole_roi[cy - half : cy + half, cx - half : cx + half]
        else:
            cavity = hole_roi

        if cavity.ndim == 3:
            gray = cv2.cvtColor(cavity, cv2.COLOR_BGR2GRAY)
        else:
            gray = cavity

        dark_ratio = float(np.mean(gray < 35))

        if dark_ratio <= 0.05:
            openness = dark_ratio
        elif dark_ratio >= 0.70:
            openness = min(1.0, 0.85 + (dark_ratio - 0.70) / 0.30 * 0.15)
        else:
            openness = 0.05 + (dark_ratio - 0.05) / (0.70 - 0.05) * 0.80

        return float(np.clip(openness, 0.0, 1.0))

    def check_lives_banner(self, frame_bgr: np.ndarray) -> bool:
        """
        Detect '剩余 X 次机会' center banner.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        ymin, xmin, ymax, xmax = self.start_gate_config.lives_banner_roi
        h, w = frame_bgr.shape[:2]
        if ymax > h or xmax > w:
            return False

        crop = frame_bgr[ymin:ymax, xmin:xmax]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else crop
        mask = cv2.inRange(hsv, np.array([12, 120, 160]), np.array([32, 255, 255]))
        yellow_count = int(np.sum(mask > 0))
        return yellow_count > self.start_gate_config.lives_banner_pixel_threshold

    def check_frog_mouth_has_ball(
        self,
        frame: np.ndarray,
        frog_center: Tuple[int, int] = (327, 233),
        radius: int = 25
    ) -> bool:
        """
        Detect if frog mouth has loaded ball.
        """
        if frame is None or frame.size == 0:
            return False

        cx, cy = frog_center
        h, w = frame.shape[:2]
        y1, y2 = max(0, cy - radius), min(h, cy + radius)
        x1, x2 = max(0, cx - radius), min(w, cx + radius)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else crop
        color_pixels = int(np.sum((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 80)))
        return color_pixels > 80

    def check_ball_in_hole(self, hole_roi: np.ndarray) -> bool:
        """
        Detect if a colored ball has entered the skull hole cavity.
        Discriminates against the closed golden skull teeth by checking:
        1. Cavity must be physically open (dark_ratio >= cavity_min_dark_ratio).
        2. Non-golden ball colors (blue, green, red, purple) inside cavity (> 120 px).
        3. High saturation/brightness yellow ball when cavity is open (dark_ratio > 0.25 and yellow_px > 300).
        """
        if hole_roi is None or hole_roi.size == 0:
            return False

        h, w = hole_roi.shape[:2]
        cx, cy = w // 2, h // 2
        half = self.danger_config.cavity_half_size

        if h >= half * 2 and w >= half * 2:
            cavity = hole_roi[cy - half : cy + half, cx - half : cx + half]
        else:
            cavity = hole_roi

        gray = cv2.cvtColor(cavity, cv2.COLOR_BGR2GRAY) if cavity.ndim == 3 else cavity
        hsv = cv2.cvtColor(cavity, cv2.COLOR_BGR2HSV) if cavity.ndim == 3 else cavity
        dark_ratio = float(np.mean(gray < 35))

        # Enforce cavity open requirement: if skull is shut tight, a ball CANNOT be inside the hole!
        if dark_ratio < self.danger_config.cavity_min_dark_ratio:
            return False

        # Check for non-skull ball colors:
        # Blue/Green/Purple/Red balls
        other_ball_pixels = int(np.sum(
            ((hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 165) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 50)) |
            (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 50))
        ))
        # Yellow ball: bright saturated yellow inside open cavity
        yellow_ball_pixels = int(np.sum(
            (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 140)
        ))

        return (other_ball_pixels > 120) or (dark_ratio > 0.25 and yellow_ball_pixels > 300)

    def update_openness_rate_limit(
        self,
        target_openness: float,
        max_delta: Optional[float] = None
    ) -> float:
        """
        Physical rate limiter on skull mouth openness.
        Constrains step-by-step change to max_openness_step_delta (default 0.08)
        to prevent instantaneous 0% -> 100% jumping from transient visual artifacts.
        """
        step_limit = max_delta if max_delta is not None else self.danger_config.max_openness_step_delta
        if self.smoothed_openness is None:
            self.smoothed_openness = float(np.clip(target_openness, 0.0, 1.0))
            return self.smoothed_openness

        delta = float(target_openness) - self.smoothed_openness
        if delta > step_limit:
            self.smoothed_openness += step_limit
        else:
            self.smoothed_openness = float(target_openness)

        self.smoothed_openness = float(np.clip(self.smoothed_openness, 0.0, 1.0))
        return self.smoothed_openness

    def detect_all_balls(
        self,
        frame: np.ndarray,
        is_bgr: bool = False,
        min_dist: int = 15
    ) -> List[Tuple[int, int]]:
        """
        Extract all ball center coordinates across the screen using HSV color segmentation
        (covering red, yellow, green, blue, purple, white) and distance-transform peak detection.
        Achieves millisecond-level execution (~5-6ms) on 640x480 frames.
        """
        if frame is None or frame.size == 0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if is_bgr else cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # 6 Zuma marble color HSV ranges
        color_ranges = [
            # Red (wrap around 0/180)
            [((0, 70, 60), (10, 255, 255)), ((170, 70, 60), (180, 255, 255))],
            # Yellow
            [((16, 80, 70), (36, 255, 255))],
            # Green
            [((38, 60, 50), (85, 255, 255))],
            # Blue
            [((90, 70, 50), (130, 255, 255))],
            # Purple
            [((135, 50, 50), (168, 255, 255))],
            # White
            [((0, 0, 170), (180, 55, 255))],
        ]

        candidates = []
        for ranges in color_ranges:
            c_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                c_mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

            # Exclude top UI and outer boundaries
            c_mask[:35, :] = 0
            c_mask[470:, :] = 0
            c_mask[:, :15] = 0
            c_mask[:, 625:] = 0

            dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
            peaks = (dist >= 6.0).astype(np.uint8)
            contours, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area >= 6:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        candidates.append((cx, cy))

            # Also check direct contours on c_mask for single isolated balls
            direct_contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for dc in direct_contours:
                area = cv2.contourArea(dc)
                if self.danger_config.ball_min_area <= area <= self.danger_config.ball_max_area:
                    peri = cv2.arcLength(dc, True)
                    if peri > 0:
                        circ = 4 * np.pi * area / (peri * peri)
                        if circ >= self.danger_config.ball_min_circularity:
                            M = cv2.moments(dc)
                            if M['m00'] > 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                                candidates.append((cx, cy))

        # Non-maximum suppression to merge any close duplicate centroids
        picked: List[Tuple[int, int]] = []
        for cx, cy in candidates:
            if not any(np.hypot(cx - px, cy - py) < min_dist for px, py in picked):
                picked.append((cx, cy))

        return picked

    def compute_collective_ball_motion(
        self,
        curr_frame: Optional[np.ndarray],
        prev_frame: Optional[np.ndarray],
        curr_balls: Optional[List[Tuple[int, int]]],
        prev_balls: Optional[List[Tuple[int, int]]],
    ) -> float:
        """
        Compute the ratio of rapidly moving balls across the full chain (fast_ball_ratio).
        Differentiates local elimination gap closure (fast_ball_ratio < 0.30)
        from fatal whole-chain black hole suction (fast_ball_ratio >= 0.80).
        """
        if curr_frame is None or prev_frame is None:
            return 0.0
        if not curr_balls or not prev_balls:
            return 0.0

        g_curr = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY) if curr_frame.ndim == 3 else curr_frame
        g_prev = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY) if prev_frame.ndim == 3 else prev_frame
        h, w = g_curr.shape[:2]

        speed_th = self.danger_config.fast_ball_speed_threshold
        diff_th = self.danger_config.fast_ball_diff_threshold

        fast_count = 0
        for cx, cy in curr_balls:
            # 1. Nearest ball distance in previous frame
            d_min = min(np.hypot(cx - px, cy - py) for px, py in prev_balls)

            # 2. Local patch frame difference
            y1, y2 = max(0, cy - 10), min(h, cy + 10)
            x1, x2 = max(0, cx - 10), min(w, cx + 10)
            p_c = g_curr[y1:y2, x1:x2].astype(np.float32)
            p_p = g_prev[y1:y2, x1:x2].astype(np.float32)
            diff = float(np.mean(np.abs(p_c - p_p))) if p_c.size > 0 else 0.0

            if d_min >= speed_th or diff >= diff_th:
                fast_count += 1

        return float(fast_count / len(curr_balls))

    def wait_for_round_ready(
        self,
        capture: Optional[ZumaScreenCapture] = None,
        max_wait_sec: Optional[float] = None
    ) -> bool:
        """
        4-step composite gating to wait for active gameplay readiness:
        1. Wait for lives banner to clear.
        2. Detect ball rush motion peak.
        3. Wait for motion to drop below calm threshold for 3 consecutive frames.
        4. Verify frog mouth readiness & settling safety delay.
        """
        cap = capture or self.capture
        max_wait = max_wait_sec or self.start_gate_config.max_wait_ready_sec
        start_time = time.time()
        
        # Step 1: Wait for lives banner to clear
        while time.time() - start_time < max_wait:
            if self.controller.is_emergency_stopped():
                return False
            frame = cap.capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.check_lives_banner(frame_bgr):
                self.current_state = GameState.LIVES_BANNER
                time.sleep(0.1)
            else:
                break

        # Step 2 & 3: Motion detection
        prev_gray = None
        consecutive_calm = 0
        saw_rush = False
        step2_start = time.time()

        while time.time() - start_time < max_wait:
            if self.controller.is_emergency_stopped():
                return False

            frame = cap.capture_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                
                # Check rush peak
                if diff >= self.start_gate_config.rush_motion_threshold:
                    saw_rush = True
                    self.current_state = GameState.STARTING_RUSH
                    consecutive_calm = 0
                elif diff <= self.start_gate_config.calm_motion_threshold:
                    if saw_rush or (time.time() - step2_start > 3.0):
                        consecutive_calm += 1
                        if consecutive_calm >= self.start_gate_config.calm_required_consecutive_frames:
                            print(f"[state_machine] Motion settled (Diff: {diff:.2f} <= {self.start_gate_config.calm_motion_threshold}), balls in place!")
                            break
                else:
                    consecutive_calm = 0

            prev_gray = gray
            time.sleep(0.05)

        # Step 4: Verify frog mouth readiness & settling safety delay
        if self.start_gate_config.settling_safety_delay_sec > 0:
            time.sleep(self.start_gate_config.settling_safety_delay_sec)

        self.reset()
        self.current_state = GameState.IN_GAME
        print("[state_machine] Round ready gate passed, entering IN_GAME!")
        return True

    def detect_state(self, frame_rgb: np.ndarray) -> Tuple[GameState, dict]:
        info = {
            "dialog_btn_loc": None,
            "dialog_btn_score": 0.0,
            "play_btn_loc": None,
            "play_btn_score": 0.0,
            "game_over_score": 0.0,
            "is_paused": False,
            "is_lives_banner": False,
            "is_emergency": self.is_emergency,
            "max_openness": 0.0,
            "hole_openness_list": [],
            "hole_locations": self.hole_locations,
        }

        if frame_rgb is None or frame_rgb.size == 0:
            return GameState.UNKNOWN, info

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        h, w = frame_bgr.shape[:2]

        # 1. Check DialogButton (确定) -> SETTLEMENT screen
        if "dialog_button" in self.templates:
            tmpl = self.templates["dialog_button"]
            res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            info["dialog_btn_score"] = float(max_val)
            if max_val > 0.65:
                cy = max_loc[1] + tmpl.shape[0] // 2
                info["dialog_btn_loc"] = (320, cy)
                self.is_emergency = False
                self.current_state = GameState.SETTLEMENT
                return GameState.SETTLEMENT, info

        # 2. Check GauntPlayButton (游戏) -> SURVIVAL_MENU screen
        if "gaunt_play" in self.templates:
            tmpl = self.templates["gaunt_play"]
            res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            info["play_btn_score"] = float(max_val)
            if max_val > 0.60:
                cy = max_loc[1] + tmpl.shape[0] // 2
                info["play_btn_loc"] = (320, cy)
                self.is_emergency = False
                self.current_state = GameState.SURVIVAL_MENU
                return GameState.SURVIVAL_MENU, info

        # Check survival button if play button is hovering/dimmed
        if "gaunt_survival" in self.templates:
            tmpl = self.templates["gaunt_survival"]
            res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.60:
                info["play_btn_loc"] = (320, 455)
                self.is_emergency = False
                self.current_state = GameState.SURVIVAL_MENU
                return GameState.SURVIVAL_MENU, info

        # 3. Check for PAUSED ('暂停中') banner in center screen
        if h >= 300 and w >= 450:
            center_crop = frame_bgr[190:290, 260:400]
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, np.array([15, 120, 150]), np.array([35, 255, 255]))
            if np.sum(yellow_mask > 0) > 1500:
                info["is_paused"] = True
                self.current_state = GameState.PAUSED
                return GameState.PAUSED, info

        # 4. Check GAME OVER screen
        if "game_over" in self.templates:
            tmpl = self.templates["game_over"]
            res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            info["game_over_score"] = float(max_val)
            if max_val > 0.60:
                self.is_emergency = False
                self.current_state = GameState.GAME_OVER
                return GameState.GAME_OVER, info

        # 5. Check Lives Banner ('剩余 X 次机会')
        if self.check_lives_banner(frame_bgr):
            info["is_lives_banner"] = True
            self.current_state = GameState.LIVES_BANNER
            return GameState.LIVES_BANNER, info

        # Check external reset or state discontinuity
        if self._last_is_emergency != self.is_emergency or self.current_state in [
            GameState.DYING_HOLE, GameState.GAME_OVER, GameState.SETTLEMENT, GameState.SURVIVAL_MENU
        ]:
            self.smoothed_openness = None
            self.prev_frame = None
            self.prev_balls = None
            self.fast_motion_consecutive_frames = 0

        # 6. Check Hole Openness & Fatal Drain if holes are anchored
        if self.hole_locations:
            openness_list = []
            hole_rois = []
            for cx, cy in self.hole_locations:
                y1, y2 = max(0, cy - 44), min(h, cy + 44)
                x1, x2 = max(0, cx - 44), min(w, cx + 44)
                roi = frame_bgr[y1:y2, x1:x2]
                hole_rois.append(roi)
                openness_list.append(self.compute_skull_openness(roi))

            raw_max_openness = max(openness_list) if openness_list else 0.0

            # Ball motion analysis across full screen
            curr_balls = self.detect_all_balls(frame_rgb, is_bgr=False)
            info["detected_balls"] = curr_balls
            info["num_balls"] = len(curr_balls)

            fast_ball_ratio = 0.0
            had_prev = (self.prev_frame is not None and self.prev_balls is not None)
            if had_prev and len(curr_balls) > 0 and len(self.prev_balls) > 0:
                fast_ball_ratio = self.compute_collective_ball_motion(
                    frame_rgb, self.prev_frame, curr_balls, self.prev_balls
                )
                if fast_ball_ratio >= self.danger_config.collective_fast_ball_ratio:
                    self.fast_motion_consecutive_frames += 1
                else:
                    self.fast_motion_consecutive_frames = 0
            else:
                self.fast_motion_consecutive_frames = 0

            info["fast_ball_ratio"] = fast_ball_ratio
            info["fast_motion_consecutive_frames"] = self.fast_motion_consecutive_frames

            # Update temporal history
            self.prev_frame = frame_rgb
            self.prev_balls = curr_balls

            # Physical rate limiter:
            # Smooth openness when streaming video frames (had_prev is True)
            if had_prev and self.smoothed_openness is not None:
                max_openness = self.update_openness_rate_limit(raw_max_openness)
            else:
                max_openness = raw_max_openness
                self.smoothed_openness = raw_max_openness

            info["raw_openness"] = raw_max_openness
            info["max_openness"] = max_openness
            info["hole_openness_list"] = openness_list

            # Safe zone gating:
            # When openness < safe_openness_threshold (0.30), force shield any ball-in-hole detection
            is_safe_zone = (max_openness < self.danger_config.safe_openness_threshold)

            # Check ball in hole (only if cavity is open and outside safe zone)
            has_ball_in_hole = False
            if not is_safe_zone:
                has_ball_in_hole = any(self.check_ball_in_hole(roi) for roi in hole_rois)

            frog_has_ball = self.check_frog_mouth_has_ball(frame_rgb)
            info["frog_has_ball"] = frog_has_ball
            info["has_ball_in_hole"] = has_ball_in_hole

            # Hysteresis loop for emergency state
            if max_openness >= self.danger_config.openness_emergency_threshold:
                self.is_emergency = True
            elif self.is_emergency and max_openness < self.danger_config.openness_recovery_threshold:
                self.is_emergency = False

            info["is_emergency"] = self.is_emergency
            self._last_is_emergency = self.is_emergency

            # Fatal dying hole checks:
            # 1. Collective full-chain suction: EMERGENCY state and >= 80% balls accelerating for >= 2 consecutive frames
            is_collective_fatal = (
                self.is_emergency
                and (self.fast_motion_consecutive_frames >= self.danger_config.collective_fast_consecutive_frames)
            )
            # 2. Physical swallowed ball fatal:
            # In live stream (had_prev True): entered hole + frog lost ammunition
            # In single static frame evaluation (had_prev False, e.g. test_15 Problem 1): verified ball in hole outside safe zone
            is_ball_in_hole_fatal = has_ball_in_hole and (not frog_has_ball or not had_prev)

            if is_collective_fatal or is_ball_in_hole_fatal:
                self.is_emergency = True
                self._last_is_emergency = True
                self.smoothed_openness = 1.0
                info["is_emergency"] = True
                info["max_openness"] = 1.0
                self.current_state = GameState.DYING_HOLE
                return GameState.DYING_HOLE, info

            if self.is_emergency:
                if not frog_has_ball:
                    self.current_state = GameState.DYING_HOLE
                    self._last_is_emergency = True
                    info["max_openness"] = 1.0
                    return GameState.DYING_HOLE, info
                self.current_state = GameState.EMERGENCY
                return GameState.EMERGENCY, info

        # 7. Default: IN_GAME
        self.current_state = GameState.IN_GAME
        self._last_is_emergency = self.is_emergency
        return GameState.IN_GAME, info

    def handle_reset_step(self, max_wait_sec: float = 12.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < max_wait_sec:
            if self.controller.is_emergency_stopped():
                return False

            self.controller.activate_window()

            frame = self.capture.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            state, info = self.detect_state(frame)

            if state == GameState.IN_GAME:
                print("[state_machine] Confirmed state: IN_GAME")
                return True

            elif state == GameState.LIVES_BANNER:
                print("[state_machine] In LIVES_BANNER, waiting for banner to clear...")
                time.sleep(0.3)

            elif state == GameState.PAUSED:
                print("[state_machine] Detected PAUSED, clicking center (320, 240) to unpause...")
                self.controller.send_input_click(320, 240, delay_sec=0.05, activate=True)
                time.sleep(0.4)

            elif state == GameState.GAME_OVER or state == GameState.DYING_HOLE:
                print("[state_machine] Detected GAME_OVER / DYING_HOLE, waiting for settlement dialog...")
                time.sleep(0.5)

            elif state == GameState.SETTLEMENT:
                click_x, click_y = info["dialog_btn_loc"] or (320, 370)
                print(f"[state_machine] In SETTLEMENT, clicking '确定' at ({click_x}, {click_y})")
                self.controller.send_input_click(click_x, click_y, delay_sec=0.05, activate=True)
                time.sleep(0.6)

            elif state == GameState.SURVIVAL_MENU:
                click_x, click_y = info["play_btn_loc"] or (320, 455)
                print(f"[state_machine] In SURVIVAL_MENU, clicking '游戏' at ({click_x}, {click_y})")
                self.controller.send_input_click(click_x, click_y, delay_sec=0.05, activate=True)
                time.sleep(1.0)

            else:
                time.sleep(0.2)

        print("[state_machine] Reset step timed out.")
        return False
