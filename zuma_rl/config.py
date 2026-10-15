from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import os

@dataclass
class WindowConfig:
    window_class: str = "MainWindow"
    window_title_keywords: List[str] = field(default_factory=lambda: ["祖玛", "zuma", "Zuma"])
    desktop_name: str = "Default"
    expected_client_width: int = 640
    expected_client_height: int = 480
    use_post_message: bool = False  # PopCap Zuma uses physical cursor polling for aiming; SendInput ensures reliable interaction
    send_input_fallback: bool = True
    auto_activate_window: bool = True
    emergency_stop_key: str = "f12"
    emergency_stop_vk: int = 0x7B  # VK_F12
    action_bounds: Optional[Tuple[int, int, int, int]] = None

    def __post_init__(self):
        if self.action_bounds is not None:
            if not (isinstance(self.action_bounds, (tuple, list)) and len(self.action_bounds) == 4):
                raise ValueError(f"action_bounds must be a 4-tuple of (min_x, min_y, max_x, max_y), got {self.action_bounds}")
            min_x, min_y, max_x, max_y = self.action_bounds
            if min_x >= max_x or min_y >= max_y:
                raise ValueError(f"action_bounds min must be less than max, got min=({min_x}, {min_y}), max=({max_x}, {max_y})")

    @property
    def playable_area(self) -> Optional[Tuple[int, int, int, int]]:
        return self.action_bounds

    @playable_area.setter
    def playable_area(self, value: Optional[Tuple[int, int, int, int]]):
        self.action_bounds = value
        self.__post_init__()

@dataclass
class EnvConfig:
    # Observation specifications
    obs_width: int = 240
    obs_height: int = 180
    num_stack_frames: int = 4
    obs_channels: int = 12  # 3 * 4
    
    # Action specifications
    grid_cols: int = 32
    grid_rows: int = 24
    num_spatial_actions: int = 768  # 32 * 24
    num_aux_actions: int = 2        # 0: Right Click (Swap), 1: No-op
    total_actions: int = 770        # 768 + 2
    
    # Safe playable action area bounds (min_x, min_y, max_x, max_y) in 640x480 client coordinates
    # Excludes top menu/status bar (y < 55) and margins (x: [20, 620], y: [55, 460])
    # to avoid clicking the "Menu" button (top-right) and window borders.
    action_bounds: Optional[Tuple[int, int, int, int]] = (20, 55, 620, 460)
    
    # Timing
    decision_hz: float = 10.0       # 10 Hz
    step_duration_sec: float = 0.1  # 100 ms
    
    # Score parsing ROI (y_start, y_end, x_start, x_end) in 640x480 client space
    # The black score window is located in the top bar between the red badge and progress bar
    score_roi: Tuple[int, int, int, int] = (2, 22, 265, 420)
    
    # Reward shaping weights
    delta_score_scale: float = 0.1  # Scale raw Zuma score increments
    step_penalty: float = -0.02     # Slight penalty per decision to discourage spam
    empty_shot_penalty: float = -0.05
    death_penalty: float = -100.0   # Large penalty for dying
    survival_reward: float = 0.01   # Small positive reward per living step
    
    # Episode limits
    max_episode_steps: int = 3000

    def __post_init__(self):
        if self.action_bounds is not None:
            if not (isinstance(self.action_bounds, (tuple, list)) and len(self.action_bounds) == 4):
                raise ValueError(f"action_bounds must be a 4-tuple of (min_x, min_y, max_x, max_y), got {self.action_bounds}")
            min_x, min_y, max_x, max_y = self.action_bounds
            if min_x >= max_x or min_y >= max_y:
                raise ValueError(f"action_bounds min must be less than max, got min=({min_x}, {min_y}), max=({max_x}, {max_y})")

    @property
    def playable_area(self) -> Optional[Tuple[int, int, int, int]]:
        return self.action_bounds

    @playable_area.setter
    def playable_area(self, value: Optional[Tuple[int, int, int, int]]):
        self.action_bounds = value
        self.__post_init__()

@dataclass
class ModelConfig:
    in_channels: int = 12
    grid_rows: int = 24
    grid_cols: int = 32
    num_aux_actions: int = 2
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 128])
    hidden_dim: int = 256

@dataclass
class TrainConfig:
    device: str = "cuda"
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 50000
    min_replay_size: int = 500
    
    # Prioritized Experience Replay (PER) parameters
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 20000
    per_eps: float = 1e-6
    
    # Epsilon greedy schedule
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 25000
    
    # Target network update
    target_update_interval: int = 1000
    
    # Logging and checkpoints
    checkpoint_dir: str = r"d:\zuma\checkpoints"
    log_interval_steps: int = 20
    checkpoint_interval_steps: int = 1000
    max_train_steps: int = 100000

@dataclass
class DangerConfig:
    # 洞口匹配模板文件名
    hole_template_name: str = "Hole.gif"
    # 洞口掩膜文件名
    hole_mask_name: str = "_Hole.gif"
    # 骷髅完全张开进入紧急状态阈值 (图5)
    openness_emergency_threshold: float = 0.85
    # 翻盘消除后骷髅重新闭合解除紧急状态阈值 (图3/4)
    openness_recovery_threshold: float = 0.70
    # 危险度密集塑形惩罚权重 (0.0 表示关闭)
    danger_penalty_weight: float = -0.05
    # 洞口中心核心腔体半宽 (像素)
    cavity_half_size: int = 20
    # 洞口绝对匹配基础阈值
    hole_match_threshold: float = 0.75
    # 多洞口锚定时相对峰值的最大容忍下降差值 (防止单轨关卡杂质纹理误识别)
    hole_match_relative_drop: float = 0.08
    # 关卡最大支持洞口数 (单轨 1, 双轨 2, 可扩展)
    max_holes: int = 2

@dataclass
class StartGateConfig:
    # 图2“剩余X次机会”屏幕中央检测区域 (ymin, xmin, ymax, xmax)
    lives_banner_roi: Tuple[int, int, int, int] = (160, 100, 280, 540)
    # 图2橙黄色大字横幅最小黄色像素命中阈值 (真实横幅约9000+, 正常局黄球<=3100)
    lives_banner_pixel_threshold: int = 5500
    # 高速冲球阶段平均灰度绝对帧差阈值 (判定冲球中)
    rush_motion_threshold: float = 12.0
    # 刹停定格平稳阶段帧差上限 (判定定格刹停)
    calm_motion_threshold: float = 3.0
    # 判定刹停所需连续维持平稳的采样帧数
    calm_required_consecutive_frames: int = 3
    # 刹停后等待青蛙装填就位的安全防抖延时 (秒)
    settling_safety_delay_sec: float = 0.20
    # 最大等待开局超时 (秒)
    max_wait_ready_sec: float = 12.0

def detect_game_dir(explicit_path: Optional[str] = None) -> str:
    """
    Intelligently find the Zuma game directory containing images_chs and fonts.
    Supports both '祖玛' and '祖玛豪华版' paths across workspace and drives.
    """
    if explicit_path and os.path.isdir(explicit_path) and os.path.isdir(os.path.join(explicit_path, "images_chs")):
        return explicit_path
    
    candidates = [
        r"d:\zuma\祖玛豪华版",
        r"d:\zuma\祖玛",
        os.path.join(os.getcwd(), "祖玛豪华版"),
        os.path.join(os.getcwd(), "祖玛"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "祖玛豪华版"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "祖玛"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isdir(os.path.join(c, "images_chs")):
            return c
            
    # Fallback to default
    return r"d:\zuma\祖玛豪华版"

@dataclass
class ZumaConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    danger: DangerConfig = field(default_factory=DangerConfig)
    start_gate: StartGateConfig = field(default_factory=StartGateConfig)
    game_dir: str = field(default_factory=detect_game_dir)

    def __post_init__(self):
        self.game_dir = detect_game_dir(self.game_dir)
