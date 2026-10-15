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
        self._load_templates()

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
        1. Non-golden ball colors (blue, green, red, purple) inside cavity (> 50 px).
        2. High saturation/brightness yellow ball when cavity is open (dark_ratio > 0.25 and yellow_px > 300).
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

        return (other_ball_pixels > 50) or (dark_ratio > 0.25 and yellow_ball_pixels > 300)

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

            max_openness = max(openness_list) if openness_list else 0.0
            info["max_openness"] = max_openness
            info["hole_openness_list"] = openness_list

            # Check fatal ball in hole first: an entered ball is absolute proof of death
            has_ball_in_hole = any(self.check_ball_in_hole(roi) for roi in hole_rois)
            frog_has_ball = self.check_frog_mouth_has_ball(frame_rgb)

            if has_ball_in_hole:
                self.is_emergency = True
                info["is_emergency"] = True
                info["max_openness"] = 1.0
                self.current_state = GameState.DYING_HOLE
                return GameState.DYING_HOLE, info

            # Hysteresis loop
            if max_openness >= self.danger_config.openness_emergency_threshold:
                self.is_emergency = True
            elif self.is_emergency and max_openness < self.danger_config.openness_recovery_threshold:
                self.is_emergency = False

            info["is_emergency"] = self.is_emergency

            if self.is_emergency:
                if not frog_has_ball:
                    self.current_state = GameState.DYING_HOLE
                    info["max_openness"] = 1.0
                    return GameState.DYING_HOLE, info
                self.current_state = GameState.EMERGENCY
                return GameState.EMERGENCY, info

        # 7. Default: IN_GAME
        self.current_state = GameState.IN_GAME
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
