import argparse
import sys
import torch
from zuma_rl.config import ZumaConfig
from zuma_rl.engine.trainer import ZumaTrainer

def main():
    parser = argparse.ArgumentParser(description="PopCap Zuma 1.0 End-to-End RL Training System")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for learner (cuda/cpu)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--buffer-capacity", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--min-replay-size", type=int, default=200, help="Warmup transitions before training begins")
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum gradient update steps")
    parser.add_argument("--checkpoint-dir", type=str, default=r"d:\zuma\checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()

    config = ZumaConfig()
    config.train.device = args.device
    config.train.learning_rate = args.lr
    config.train.batch_size = args.batch_size
    config.train.replay_capacity = args.buffer_capacity
    config.train.min_replay_size = args.min_replay_size
    config.train.max_train_steps = args.max_steps
    config.train.checkpoint_dir = args.checkpoint_dir

    trainer = ZumaTrainer(config)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.learner.agent.load_checkpoint(args.resume)

    trainer.start()

if __name__ == "__main__":
    main()
