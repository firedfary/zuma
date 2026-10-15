import ctypes
from ctypes import wintypes
import time
from typing import Optional, Tuple
import win32con
from ..config import WindowConfig, EnvConfig
from .capture import ensure_default_desktop, find_zuma_window

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Configure 64-bit Windows ctypes argtypes and restypes to prevent handle truncation
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.SetWindowPos.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = wintypes.SHORT

class ZumaInputController:
    def __init__(self, hwnd: Optional[int] = None, window_config: Optional[WindowConfig] = None, env_config: Optional[EnvConfig] = None):
        self.w_config = window_config or WindowConfig()
        self.e_config = env_config or EnvConfig()
        ensure_default_desktop(self.w_config.desktop_name)
        self.hwnd = hwnd or find_zuma_window()
        self.stop_requested = False
        if self.w_config.auto_activate_window and self.hwnd:
            self.activate_window()

    def update_hwnd(self) -> Optional[int]:
        self.hwnd = find_zuma_window()
        return self.hwnd

    def activate_window(self) -> bool:
        """
        Brings Zuma window to top and ensures interactive focus.
        """
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            self.update_hwnd()
            if not self.hwnd:
                return False
        ensure_default_desktop(self.w_config.desktop_name)
        user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
        # Briefly topmost then non-topmost brings window to foreground without locking Z-order
        user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 1 | 2 | 0x40)
        user32.SetWindowPos(self.hwnd, -2, 0, 0, 0, 0, 1 | 2 | 0x40)
        user32.SetForegroundWindow(self.hwnd)
        return True

    @staticmethod
    def _make_lparam(x: int, y: int) -> int:
        return ((y & 0xFFFF) << 16) | (x & 0xFFFF)

    def is_emergency_stopped(self) -> bool:
        """
        Check if the F12 hotkey is pressed to trigger an emergency stop.
        """
        if self.stop_requested:
            return True
        # GetAsyncKeyState returns MSB set if key is down
        state = user32.GetAsyncKeyState(self.w_config.emergency_stop_vk)
        if state & 0x8000:
            print("[input] EMERGENCY STOP TRIGGERED VIA F12!")
            self.stop_requested = True
            return True
        return False

    def get_client_size(self) -> Tuple[int, int]:
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            return self.w_config.expected_client_width, self.w_config.expected_client_height
        rect = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return self.w_config.expected_client_width, self.w_config.expected_client_height
        return w, h

    def get_screen_pos(self) -> Tuple[int, int]:
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            return 0, 0
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return pt.x, pt.y

    def post_left_click(self, client_x: int, client_y: int, delay_sec: float = 0.02):
        """
        Send left-click using PostMessage to the Zuma window.
        """
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            self.update_hwnd()
            if not self.hwnd:
                return

        lparam = self._make_lparam(int(client_x), int(client_y))
        user32.PostMessageW(self.hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
        user32.PostMessageW(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
        if delay_sec > 0:
            time.sleep(delay_sec)
        user32.PostMessageW(self.hwnd, win32con.WM_LBUTTONUP, 0, lparam)

    def post_right_click(self, client_x: int = 320, client_y: int = 240, delay_sec: float = 0.02):
        """
        Send right-click (ball swap) using PostMessage to the Zuma window.
        """
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            self.update_hwnd()
            if not self.hwnd:
                return

        lparam = self._make_lparam(int(client_x), int(client_y))
        user32.PostMessageW(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, lparam)
        if delay_sec > 0:
            time.sleep(delay_sec)
        user32.PostMessageW(self.hwnd, win32con.WM_RBUTTONUP, 0, lparam)

    def send_input_click(self, client_x: int, client_y: int, delay_sec: float = 0.02, activate: bool = False):
        """
        Left-click via physical mouse event at screen coordinates.
        """
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            self.update_hwnd()
            if not self.hwnd:
                return
        ensure_default_desktop(self.w_config.desktop_name)
        if activate and self.w_config.auto_activate_window:
            self.activate_window()
        sx, sy = self.get_screen_pos()
        target_x = sx + int(client_x)
        target_y = sy + int(client_y)
        user32.SetCursorPos(target_x, target_y)
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        if delay_sec > 0:
            time.sleep(delay_sec)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def send_input_right_click(self, client_x: int = 320, client_y: int = 240, delay_sec: float = 0.02, activate: bool = False):
        """
        Right-click (ball swap) via physical mouse event at screen coordinates.
        """
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            self.update_hwnd()
            if not self.hwnd:
                return
        ensure_default_desktop(self.w_config.desktop_name)
        if activate and self.w_config.auto_activate_window:
            self.activate_window()
        sx, sy = self.get_screen_pos()
        target_x = sx + int(client_x)
        target_y = sy + int(client_y)
        user32.SetCursorPos(target_x, target_y)
        MOUSEEVENTF_RIGHTDOWN = 0x0008
        MOUSEEVENTF_RIGHTUP = 0x0010
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        if delay_sec > 0:
            time.sleep(delay_sec)
        user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def get_action_coords(self, action_idx: int) -> Tuple[str, Tuple[int, int]]:
        """
        Calculate target coordinates for an action index without executing clicks:
        - 0 to (grid_cols * grid_rows - 1): Spatial click on grid cell center within safe action bounds
        - num_spatial_actions (768): Ball swap (Right Click) at window center
        - num_spatial_actions + 1 (769): No-op (0, 0)
        Returns:
            action_type: "click", "swap", or "noop"
            coords: (client_x, client_y)
        """
        if action_idx < 0 or action_idx >= self.e_config.total_actions:
            raise ValueError(f"Invalid action_idx: {action_idx}. Must be in range [0, {self.e_config.total_actions - 1}].")

        num_spatial = self.e_config.num_spatial_actions
        cw, ch = self.get_client_size()

        if 0 <= action_idx < num_spatial:
            row = action_idx // self.e_config.grid_cols
            col = action_idx % self.e_config.grid_cols

            bounds = self.w_config.action_bounds if self.w_config.action_bounds is not None else self.e_config.action_bounds
            if bounds is None:
                exp_w = self.w_config.expected_client_width
                exp_h = self.w_config.expected_client_height
                bounds = (0, 0, exp_w, exp_h)

            if not (isinstance(bounds, (tuple, list)) and len(bounds) == 4):
                raise ValueError(f"action_bounds must be a 4-tuple of (min_x, min_y, max_x, max_y), got {bounds}")
            min_x, min_y, max_x, max_y = bounds
            if min_x >= max_x or min_y >= max_y:
                raise ValueError(f"action_bounds min must be less than max, got min=({min_x}, {min_y}), max=({max_x}, {max_y})")

            exp_w = self.w_config.expected_client_width
            exp_h = self.w_config.expected_client_height
            scale_x = cw / exp_w if exp_w > 0 else 1.0
            scale_y = ch / exp_h if exp_h > 0 else 1.0

            eff_min_x = min_x * scale_x
            eff_max_x = max_x * scale_x
            eff_min_y = min_y * scale_y
            eff_max_y = max_y * scale_y

            cell_w = (eff_max_x - eff_min_x) / self.e_config.grid_cols
            cell_h = (eff_max_y - eff_min_y) / self.e_config.grid_rows

            raw_px = eff_min_x + (col + 0.5) * cell_w
            raw_py = eff_min_y + (row + 0.5) * cell_h

            px = int(max(eff_min_x, min(eff_max_x, raw_px)))
            py = int(max(eff_min_y, min(eff_max_y, raw_py)))
            return "click", (px, py)

        elif action_idx == num_spatial:
            cx, cy = cw // 2, ch // 2
            return "swap", (cx, cy)

        elif action_idx == num_spatial + 1:
            return "noop", (0, 0)

        else:
            raise ValueError(f"Invalid action_idx: {action_idx}. Must be in range [0, {self.e_config.total_actions - 1}].")

    def execute_action(self, action_idx: int, dry_run: bool = False) -> Tuple[str, Tuple[int, int]]:
        """
        Execute an RL action index:
        - 0 to (grid_cols * grid_rows - 1): Spatial click on grid cell center within safe action bounds
        - num_spatial_actions (768): Ball swap (Right Click)
        - num_spatial_actions + 1 (769): No-op (Wait)
        Args:
            action_idx: RL action integer
            dry_run: If True, calculates coordinates without sending physical mouse events
        Returns:
            action_type: "click", "swap", or "noop"
            coords: (client_x, client_y)
        """
        if self.is_emergency_stopped():
            return "emergency_stopped", (0, 0)

        act_type, coords = self.get_action_coords(action_idx)
        px, py = coords

        if not dry_run:
            if act_type == "click":
                if self.w_config.use_post_message:
                    self.post_left_click(px, py)
                else:
                    self.send_input_click(px, py)
            elif act_type == "swap":
                if self.w_config.use_post_message:
                    self.post_right_click(px, py)
                else:
                    self.send_input_right_click(px, py)

        return act_type, coords
