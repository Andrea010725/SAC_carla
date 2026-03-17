"""Minimal PPO entry for the RL planner on four scenarios only."""
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def train_ppo():
    from train_ppo_with_wandb import train_ppo as train_ppo_unified

    print("[Route] train_ppo_minimal_4scenarios -> train_ppo_with_wandb(profile='minimal4')")
    return train_ppo_unified(profile="minimal4")


if __name__ == "__main__":
    model_path = train_ppo()
    print("\nNext step: use the trained PPO planner")
    print(f"  ppo_model_path='{model_path}'")
