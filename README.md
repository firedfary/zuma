# Zuma 1.0 (祖玛豪华版) 端到端强化学习训练系统

本系统是针对经典游戏《祖玛1.0》（PopCap Zuma）设计与实现的端到端强化学习（End-to-End Reinforcement Learning）完整训练与自动化控制系统。

- 📐 **系统架构设计详述**: 参见 [ARCHITECTURE.md](file:///d:/zuma/ARCHITECTURE.md)
- 📈 **工程进度与缺陷复盘**: 参见 [PROGRESS.md](file:///d:/zuma/PROGRESS.md)

---

## 🌟 系统技术架构与核心特性

### 1. 空间全卷积 Q 网络 (Spatial FCN-DQN / Rainbow)
- **观测输入**: 游戏画面 RGB 适度降采样至 `240x180`，4 帧历史堆叠（`12` 通道输入），保留球串运动时序信息。
- **空间动作热力图 (Spatial Action Map)**: 骨干网络通过多层全卷积与双线性插值输出 `32x24` 网格空间 Q 值图（共 768 个屏幕网格点击落点），保持二维空间几何拓扑一致性。
- **辅助动作分支 (Auxiliary Action Branch)**: 独立分支输出右键换球（Swap Ball）与观望不操作（No-op）两个辅助动作，总动作空间为 `770` 维。
- **Dueling 架构**: 状态价值流 $V(s)$ 与空间/辅助优势流 $A(s, a)$ 独立分离，提升值函数评估稳定性与采样效率。
- **Double DQN**: 解耦动作选择与价值评估，彻底消除 Q 值过高估计偏差。

### 2. 异步双线程/进程高吞吐并发架构
- **Actor 工作流 (10 Hz 无阻塞交互)**:
  - 严控 100ms（10 Hz）决策步长，高保真采集真实环境交互数据。
  - 将 `(state, action, reward, next_state, done)` 实时压入经验回放池。
  - 周期性从 Learner 同步最新网络权重。
- **Learner 工作流 (独立 GPU 梯度更新)**:
  - 独立运行于 GPU（Tesla V100 等），免受环境画面抓取和步频休眠影响。
  - 单卡训练吞吐量可达 80~100 steps/s。
  - 周期性硬同步 Target 网络，并自动导出检查点（Checkpoint）。

### 3. 经验回放池 (Prioritized Experience Replay, PER)
- 基于二叉树 **SumTree** 实现 $O(\log N)$ 时间复杂度的优先采样与 TD 误差优先级更新。
- 具备重要性采样权重校正（Importance Sampling Weights），退火调整 $\beta$ 超参数。
- 线程安全锁保护，支持并发推流与抽样。

### 4. 环境交互与重置状态机自动化
- **GDI BitBlt / PrintWindow 后台抓取**: 跨 Windows 虚拟桌面（Default Desktop）无黑屏抓取游戏客户区画面。
- **消息投递控制**: 优先使用 Windows `PostMessage` 发送鼠标点击与右键换球，不强占物理鼠标指针，辅以 `SendInput` 保底。
- **模板匹配驱动的四态循环状态机**:
  - `IN_GAME`: 游戏中正常状态，10 Hz 决策控制。
  - `GAME_OVER`: 检测到中心大字体 "GAME OVER" 画面，立刻中止操作，进入结算倒计时。
  - `SETTLEMENT`: 结算弹窗画面，自动定位并点击“确定”按钮（DialogButton）。
  - `SURVIVAL_MENU`: 生存模式主界面，自动定位并点击正下方“游戏”按钮（GauntPlayButton）开启下一轮训练。
- **金色数字模板匹配解析得分 (Cancun13 Font OCR)**:
  - 采用祖玛官方点阵字体 Cancun13 模板进行非极大值抑制（NMS）匹配，精准无误解析游戏上方金色数字分值。
- **复合奖励塑形 (Composite Reward Shaping)**:
  - 得分增量奖励: $\Delta\text{Score} \times \text{scale}$
  - 步频微惩罚: $-0.02$（避免无休止挂机）
  - 空枪惩罚: 点击未得分时施加轻微惩罚 $-0.05$（引导精准击球）
  - 存活奖励: $+0.01$
  - 死亡惩罚: 触发 Game Over 时 $-100.0$

### 5. 全局急停机制
- 任何时候按下键盘 **F12** 键，系统立即无条件安全熔断并退出，保护系统与游戏进度。

---

## 📁 目录结构

```
d:\zuma\
├── zuma_rl\                    # 强化学习系统核心包
│   ├── config.py               # 整体数据类配置（窗口、环境、网络、训练）
│   ├── window\
│   │   ├── capture.py          # Windows GDI / PrintWindow 屏幕抓取
│   │   └── input_controller.py # PostMessage / SendInput 模拟输入与 F12 监控
│   ├── env\
│   │   ├── digit_reader.py     # 祖玛 Cancun13 点阵字体得分匹配解析
│   │   ├── state_machine.py    # 游戏结束与生存模式自动复位状态机
│   │   └── zuma_gym_env.py     # Gymnasium 标准环境封装（10Hz 步频调节与奖励塑形）
│   ├── model\
│   │   └── spatial_fcn_dqn.py  # 空间全卷积 Q 网络 (Dueling Spatial FCN-DQN)
│   ├── buffer\
│   │   └── prioritized_replay.py # 线程安全 SumTree 优先经验回放池 (PER)
│   ├── agent\
│   │   └── dqn_agent.py        # Double DQN 算法逻辑、损失计算与检查点存取
│   └── engine\
│       ├── actor.py            # 10 Hz 交互推流线程
│       ├── learner.py          # 独立 GPU 训练更新线程
│       └── trainer.py          # 并发调度与优雅退出协调器
├── train.py                    # 训练主入口脚本
├── evaluate.py                 # 模型评估与推理执行脚本
├── calibrate.py                # 视窗、OCR、状态机与动作空间校准诊断工具
├── test_suite.py               # 单元与系统级全自动化测试套件
├── ARCHITECTURE.md             # 系统完整技术架构与数学设计文档
├── PROGRESS.md                 # 里程碑状态、缺陷复盘与演进路线
└── checkpoints\                # 训练模型检查点保存目录
```

---

## 🚀 快速启动指南

### 1. 运行环境
使用 conda 的 `nc` 环境运行：
```powershell
# 激活环境后执行或直接调用 nc 的 python 解释器
& "C:\Users\fired\anaconda3\envs\nc\python.exe" train.py
```

### 2. 视窗与感知系统校准诊断
在开启训练前，可先运行校准工具测试游戏窗口定位、画面抓取、数字 OCR 解析与按键映射：
```powershell
& "C:\Users\fired\anaconda3\envs\nc\python.exe" calibrate.py
```

### 3. 开启端到端强化学习训练
启动并发训练（Actor 实时打游戏采集数据，Learner GPU 加速梯度反向传播）：
```powershell
& "C:\Users\fired\anaconda3\envs\nc\python.exe" train.py --batch-size 64 --lr 0.0001 --device cuda
```
常用参数：
- `--device`: `cuda` 或 `cpu`（默认自动检测 GPU）
- `--batch-size`: 训练批次大小（默认 64）
- `--lr`: 学习率（默认 1e-4）
- `--buffer-capacity`: 经验池容量（默认 50000）
- `--min-replay-size`: 开始梯度更新前收集的步数预热阈值（默认 200）
- `--max-steps`: 最大训练步数（默认 100000）
- `--resume <checkpoint_path>`: 从已有检查点继续训练

### 4. 评估推理与展示
使用训练好的权重进行实际运行：
```powershell
& "C:\Users\fired\anaconda3\envs\nc\python.exe" evaluate.py --checkpoint d:\zuma\checkpoints\zuma_dqn_final.pt --episodes 5
```

### 5. 运行完整单元与集成测试
```powershell
& "C:\Users\fired\anaconda3\envs\nc\python.exe" test_suite.py
```
全部 10 项测试涵盖模型前向/反向、SumTree/PER 采样更新、DQN 训练步与权重存取、Cancun13 数字识别、状态机识别、动作空间映射与多线程并发安全。
