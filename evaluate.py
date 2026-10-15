import argparse
import time
import torch
import numpy as np

from zuma_rl.config import ZumaConfig
from zuma_rl.env.zuma_gym_env import ZumaGymEnv
from zuma_rl.agent.dqn_agent import ZumaDQNAgent

def evaluate(checkpoint_path: str, num_episodes: int = 5, device: str = "cuda"):
    config = ZumaConfig()
    config.train.device = device if torch.cuda.is_available() else "cpu"

    env = ZumaGymEnv(config)
    agent = ZumaDQNAgent(
        model_config=config.model,
        train_config=config.train,
        device=config.train.device
    )

    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        agent.load_checkpoint(checkpoint_path)
    else:
        print("Warning: Running evaluation with randomly initialized model.")

    print("=" * 60)
    print(f"Starting Evaluation for {num_episodes} episodes...")
    print("Press F12 for emergency stop.")
    print("=" * 60)

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            if env.controller.is_emergency_stopped():
                print("[eval] Emergency stop triggered!")
                env.close()
                return

            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(
            f"Episode {ep}/{num_episodes} Completed | "
            f"Steps: {steps} | "
            f"Score Gain: {step_info.get('episode_score_gain', 0)} | "
            f"Total Reward: {total_reward:.2f}"
        )

    env.close()
    print("Evaluation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Zuma RL agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda/cpu)")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.device)
