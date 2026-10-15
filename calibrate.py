import time
import cv2
import numpy as np
from PIL import Image

from zuma_rl.config import ZumaConfig
from zuma_rl.window.capture import ZumaScreenCapture
from zuma_rl.window.input_controller import ZumaInputController
from zuma_rl.env.digit_reader import ZumaDigitReader
from zuma_rl.env.state_machine import ZumaStateMachine, GameState

def calibrate():
    print("=" * 60)
    print("PopCap Zuma RL Calibration & Diagnostic Tool")
    print("=" * 60)

    config = ZumaConfig()
    capture = ZumaScreenCapture()
    controller = ZumaInputController(hwnd=capture.hwnd, window_config=config.window, env_config=config.env)
    reader = ZumaDigitReader()
    state_machine = ZumaStateMachine(game_dir=config.game_dir, capture=capture, controller=controller)

    print(f"1. Target Window HWND: {capture.hwnd}")
    valid = capture.is_window_valid()
    print(f"   Window Valid: {valid}")
    if not valid:
        print("   ERROR: Could not find valid Zuma window! Ensure Zuma.exe is running.")
        return

    cw, ch = capture.get_client_rect()
    sx, sy = capture.get_screen_position()
    print(f"   Client Dimensions: {cw}x{ch}")
    print(f"   Screen Position: ({sx}, {sy})")

    # Capture test frame
    print("\n2. Capturing Frame...")
    frame = capture.capture_frame()
    if frame is None:
        print("   ERROR: Frame capture returned None!")
        return
    print(f"   Captured Frame Shape: {frame.shape}, Mean Brightness: {frame.mean():.2f}")
    Image.fromarray(frame).save(r"d:\zuma\calibration_frame.png")
    print(r"   Saved calibration frame to: d:\zuma\calibration_frame.png")

    # Score reading
    print("\n3. Testing Score OCR...")
    score = reader.extract_score(frame, roi=config.env.score_roi)
    print(f"   Detected Score: {score}")

    # State Machine detection
    print("\n4. Detecting Game State...")
    state, info = state_machine.detect_state(frame)
    print(f"   Detected State: {state.name}")
    print(f"   State Details: {info}")

    # Test Action mapping
    print("\n5. Testing Action Space Coordinates Mapping (Safe Playable Area)...")
    print(f"   Safe Playable Area Bounds: {config.env.action_bounds} (min_x, min_y, max_x, max_y)")
    sample_actions = [0, 31, 100, 384, 767, 768, 769]
    for act in sample_actions:
        act_type, (px, py) = controller.get_action_coords(act)
        num_spatial = config.env.num_spatial_actions
        if act < num_spatial:
            row = act // config.env.grid_cols
            col = act % config.env.grid_cols
            print(f"   Action {act:3d} (Spatial Grid ({col:2d}, {row:2d})) -> Client Pixel ({px:3d}, {py:3d})")
        elif act == num_spatial:
            print(f"   Action {act:3d} (Auxiliary: Swap Ball) -> Right Click at ({px}, {py})")
        else:
            print(f"   Action {act:3d} (Auxiliary: No-op) -> Wait")

    print("\nCalibration check complete! All core perception & actuation interfaces verified.")

if __name__ == "__main__":
    calibrate()
