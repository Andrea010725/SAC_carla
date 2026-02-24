#!/usr/bin/env python3
"""Plot training_log.json produced by TrainingLogger.

Usage:
  python tools/plot_training_log.py training_log.json
  python tools/plot_training_log.py training_log.json out.png

If out.png is omitted, saves to training_log_plot.png in the same directory as the log.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def rolling_mean(x, window):
    if window <= 1:
        return x
    out = []
    acc = 0.0
    q = []
    for v in x:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_training_log.py training_log.json [out.png]")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else log_path.parent / "training_log_plot.png"

    with log_path.open() as f:
        data = json.load(f)

    episodes = data.get("episodes", []) if isinstance(data, dict) else data
    if not episodes:
        print("No episode data found.")
        sys.exit(1)

    ep = [e.get("episode", i + 1) for i, e in enumerate(episodes)]
    reward = [e.get("reward", 0.0) for e in episodes]
    length = [e.get("length", 0) for e in episodes]
    avg_speed = [e.get("avg_speed", 0.0) for e in episodes]
    collision_rate = [e.get("collision_rate", 0.0) for e in episodes]
    policy_loss = [e.get("policy_loss", 0.0) for e in episodes]
    value_loss = [e.get("value_loss", 0.0) for e in episodes]
    entropy = [e.get("entropy", 0.0) for e in episodes]

    # Rolling mean window: 20 (adjust if needed)
    W = 20

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].plot(ep, reward, alpha=0.35, label="reward")
    axes[0].plot(ep, rolling_mean(reward, W), label=f"reward (mean {W})")
    axes[0].set_title("Episode Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, avg_speed, alpha=0.35, label="avg_speed")
    axes[1].plot(ep, rolling_mean(avg_speed, W), label=f"avg_speed (mean {W})")
    axes[1].set_title("Average Speed")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(ep, length, alpha=0.35, label="length")
    axes[2].plot(ep, rolling_mean(length, W), label=f"length (mean {W})")
    axes[2].set_title("Episode Length")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(ep, collision_rate, alpha=0.35, label="collision_rate")
    axes[3].plot(ep, rolling_mean(collision_rate, W), label=f"collision_rate (mean {W})")
    axes[3].set_title("Collision Rate (%)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(ep, policy_loss, alpha=0.35, label="policy_loss")
    axes[4].plot(ep, rolling_mean(policy_loss, W), label=f"policy_loss (mean {W})")
    axes[4].set_title("Policy Loss")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    axes[5].plot(ep, value_loss, alpha=0.35, label="value_loss")
    axes[5].plot(ep, rolling_mean(value_loss, W), label=f"value_loss (mean {W})")
    axes[5].plot(ep, entropy, alpha=0.25, label="entropy")
    axes[5].set_title("Value Loss / Entropy")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    fig.suptitle("Training Log Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
