import os
import time
from enum import Enum, auto
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from ..config import WindowConfig, EnvConfig
from ..window.capture import ZumaScreenCapture
from ..window.input_controller import ZumaInputController

class GameState(Enum):
    UNKNOWN = auto()
    IN_GAME = auto()
    PAUSED = auto()
    GAME_OVER = auto()
    SETTLEMENT = auto()
    SURVIVAL_MENU = auto()

class ZumaStateMachine:
    """
    State machine for Zuma gameplay detection and automatic reset loop:
    - IN_GAME: Normal gameplay
    - PAUSED: Game paused with '暂停中' -> click center (320, 240) to unpause
    - GAME_OVER: Detected 'GAME OVER' screen (图2) -> halt gameplay actions, wait for settlement
    - SETTLEMENT: Dialog screen with '确定' (DialogButton) (图3) -> auto-click '确定' at (320, 370)
    - SURVIVAL_MENU: Survival/Gauntlet screen with '游戏' (GauntPlayButton) (图4) -> auto-click '游戏' at (320, 455)
    """
    def __init__(
        self,
        game_dir: str = r"d:\zuma\祖玛豪华版",
        capture: Optional[ZumaScreenCapture] = None,
        controller: Optional[ZumaInputController] = None
    ):
        self.game_dir = game_dir
        self.capture = capture or ZumaScreenCapture()
        self.controller = controller or ZumaInputController(hwnd=self.capture.hwnd)
        
        self.current_state = GameState.UNKNOWN
        self.templates = {}
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

    def detect_state(self, frame_rgb: np.ndarray) -> Tuple[GameState, dict]:
        """
        Detect current game state from frame.
        Returns:
            state: GameState enum
            info: dict with match scores and detected button coordinates
        """
        info = {
            "dialog_btn_loc": None,
            "dialog_btn_score": 0.0,
            "play_btn_loc": None,
            "play_btn_score": 0.0,
            "game_over_score": 0.0,
            "is_paused": False,
        }

        if frame_rgb is None or frame_rgb.size == 0:
            return GameState.UNKNOWN, info

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        h, w = frame_bgr.shape[:2]

        # 1. Check DialogButton (确定) -> SETTLEMENT screen (图3)
        if "dialog_button" in self.templates:
            tmpl = self.templates["dialog_button"]
            res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            info["dialog_btn_score"] = float(max_val)
            if max_val > 0.65:
                # The green DialogButton is horizontally centered at x=320, vertical center at max_loc[1] + tmpl_h//2
                cy = max_loc[1] + tmpl.shape[0] // 2
                info["dialog_btn_loc"] = (320, cy)
                self.current_state = GameState.SETTLEMENT
                return GameState.SETTLEMENT, info

        # 2. Check GauntPlayButton (游戏) -> SURVIVAL_MENU screen (图4)
        if "gaunt_play" in self.templates:
            tmpl = self.templates["gaunt_play"]
            res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            info["play_btn_score"] = float(max_val)
            if max_val > 0.60:
                cx = max_loc[0] + tmpl.shape[1] // 2
                cy = max_loc[1] + tmpl.shape[0] // 2
                info["play_btn_loc"] = (320, cy)
                self.current_state = GameState.SURVIVAL_MENU
                return GameState.SURVIVAL_MENU, info

        # Check survival button if play button is hovering/dimmed
        if "gaunt_survival" in self.templates:
            tmpl = self.templates["gaunt_survival"]
            res = cv2.matchTemplate(frame_bgr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.60:
                info["play_btn_loc"] = (320, 455)
                self.current_state = GameState.SURVIVAL_MENU
                return GameState.SURVIVAL_MENU, info

        # 3. Check for PAUSED ('暂停中') banner in center screen
        # Large yellow/gold text appears at [190:290, 260:400] when window loses focus
        if h >= 300 and w >= 450:
            center_crop = frame_bgr[190:290, 260:400]
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, np.array([15, 120, 150]), np.array([35, 255, 255]))
            if np.sum(yellow_mask > 0) > 1500:
                info["is_paused"] = True
                self.current_state = GameState.PAUSED
                return GameState.PAUSED, info

        # 4. Check GAME OVER screen (图2)
        if "game_over" in self.templates:
            tmpl = self.templates["game_over"]
            res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            info["game_over_score"] = float(max_val)
            if max_val > 0.60:
                self.current_state = GameState.GAME_OVER
                return GameState.GAME_OVER, info

        # 5. Default: If active gameplay screen elements exist, we are actively IN_GAME
        self.current_state = GameState.IN_GAME
        return GameState.IN_GAME, info

    def handle_reset_step(self, max_wait_sec: float = 12.0) -> bool:
        """
        Executes the reset state machine transitions:
        - If PAUSED: activate window and click center to unpause
        - If GAME_OVER: wait for settlement dialog
        - If SETTLEMENT: click '确定' button at (320, 370)
        - If SURVIVAL_MENU: click '游戏' button at (320, 455)
        - Loop until IN_GAME is reached or timeout.
        Returns:
            True if successfully returned to IN_GAME, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_sec:
            if self.controller.is_emergency_stopped():
                return False

            # Ensure window is active and foreground
            self.controller.activate_window()

            frame = self.capture.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            state, info = self.detect_state(frame)

            if state == GameState.IN_GAME:
                print("[state_machine] Confirmed state: IN_GAME")
                return True

            elif state == GameState.PAUSED:
                print("[state_machine] Detected PAUSED, clicking center (320, 240) to unpause...")
                self.controller.send_input_click(320, 240, delay_sec=0.05, activate=True)
                time.sleep(0.4)

            elif state == GameState.GAME_OVER:
                print("[state_machine] Detected GAME_OVER, waiting for settlement dialog...")
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
