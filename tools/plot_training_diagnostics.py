"""
训练诊断可视化脚本（本地生成图片）

功能：
1) 读取 training_log.json（episode 级别）
2) 尝试读取 wandb offline run 的 history（step 级别）
3) 输出多张诊断图，便于判断是否真正收敛

用法示例：
  python tools/plot_training_diagnostics.py
  python tools/plot_training_diagnostics.py --training-log ./training_log.json
  python tools/plot_training_diagnostics.py --wandb-run ./wandb/latest-run
"""
from __future__ import annotations

import argparse
import json
import os
import glob
from typing import Dict, List, Any

# ✅ 避免 matplotlib 缓存目录不可写
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# 工具函数
# ---------------------------
def _moving_avg(arr: List[float], w: int = 20) -> List[float]:
    out = []
    for i in range(len(arr)):
        s = max(0, i - w + 1)
        out.append(sum(arr[s:i + 1]) / (i - s + 1))
    return out


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ---------------------------
# 读取 training_log.json
# ---------------------------
def load_training_log(path: str) -> Dict[str, List[float]]:
    if not os.path.exists(path):
        return {}

    with open(path, "r") as f:
        data = json.load(f)
    eps = data.get("episodes", [])
    if not eps:
        return {}

    series = {
        "episode": [e.get("episode", i + 1) for i, e in enumerate(eps)],
        "reward": [e.get("reward", 0.0) for e in eps],
        "length": [e.get("length", 0) for e in eps],
        "avg_speed": [e.get("avg_speed", 0.0) for e in eps],
        "collision_rate": [e.get("collision_rate", 0.0) for e in eps],
        "policy_loss": [e.get("policy_loss", 0.0) for e in eps],
        "value_loss": [e.get("value_loss", 0.0) for e in eps],
        "entropy": [e.get("entropy", 0.0) for e in eps],
    }
    return series


# ---------------------------
# 读取 wandb history（offline）
# ---------------------------
def find_wandb_history(run_dir: str) -> str | None:
    # 允许传入 wandb/latest-run 或 offline-run-xxxx
    candidates = [
        os.path.join(run_dir, "files", "wandb-history.jsonl"),
        os.path.join(run_dir, "files", "wandb-history.json"),
        os.path.join(run_dir, "wandb-history.jsonl"),
        os.path.join(run_dir, "wandb-history.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 如果传入的是 wandb 目录，尝试 latest-run
    if os.path.isdir(run_dir):
        latest = os.path.join(run_dir, "latest-run")
        if os.path.islink(latest) or os.path.isdir(latest):
            return find_wandb_history(os.path.realpath(latest))
    return None


def load_wandb_history(path: str) -> Dict[str, List[float]]:
    """
    只抽取我们关心的 key，避免内存过大
    """
    keys = [
        "env_step",
        "step/reward",
        "step/speed",
        "step/y_ref",
        "debug/std_throttle",
        "debug/std_steer",
        "debug/std_y_ref",
        "debug_reward/r_progress",
        "debug_reward/r_speed",
        "debug_reward/r_obs_clear",
        "debug_reward/r_lane",
        "debug_reward/r_danger",
        "debug_reward/r_collision",
    ]
    series = {k: [] for k in keys}

    if not path or not os.path.exists(path):
        return {}

    # wandb-history.jsonl 是逐行 JSON
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for k in keys:
                if k in row:
                    series[k].append(_safe_float(row[k]))
                else:
                    series[k].append(None)

    # 如果没有 env_step，就用 index 代替
    if series.get("env_step") and all(v is None for v in series["env_step"]):
        series["env_step"] = list(range(len(series["step/reward"])))

    return series


# ---------------------------
# 绘图
# ---------------------------
def plot_episode_metrics(series: Dict[str, List[float]], out_dir: str) -> str | None:
    if not series:
        return None

    x = series["episode"]
    reward = series["reward"]
    length = series["length"]
    speed = series["avg_speed"]
    collision = series["collision_rate"]
    policy_loss = series["policy_loss"]
    value_loss = series["value_loss"]
    entropy = series["entropy"]

    ma_reward = _moving_avg(reward, 20)
    ma_speed = _moving_avg(speed, 20)
    ma_collision = _moving_avg(collision, 20)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].plot(x, reward, alpha=0.3, label="reward")
    axes[0].plot(x, ma_reward, label="reward_ma20")
    axes[0].set_title("Episode Reward")
    axes[0].legend()

    axes[1].plot(x, length, label="length")
    axes[1].set_title("Episode Length")

    axes[2].plot(x, speed, alpha=0.3, label="avg_speed")
    axes[2].plot(x, ma_speed, label="speed_ma20")
    axes[2].set_title("Avg Speed (m/s)")
    axes[2].legend()

    axes[3].plot(x, collision, alpha=0.3, label="collision_rate")
    axes[3].plot(x, ma_collision, label="collision_rate_ma20")
    axes[3].set_title("Collision Rate (%)")
    axes[3].legend()

    axes[4].plot(x, policy_loss, label="policy_loss")
    axes[4].set_title("Policy Loss")

    axes[5].plot(x, value_loss, label="value_loss")
    axes[5].set_title("Value Loss")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "episode_metrics.png")
    plt.savefig(out_path, dpi=150)
    return out_path


def plot_step_metrics(series: Dict[str, List[float]], out_dir: str) -> List[str]:
    if not series:
        return []

    env_step = series.get("env_step") or list(range(len(series["step/reward"])))

    # 1) step reward & speed
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(env_step, series.get("step/reward", []), alpha=0.3, label="step/reward")
    ax.plot(env_step, _moving_avg([_safe_float(x) for x in series.get("step/reward", [])], 200),
            label="step/reward_ma200")
    ax.plot(env_step, series.get("step/speed", []), alpha=0.3, label="step/speed")
    ax.set_title("Step Reward / Speed")
    ax.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out1 = os.path.join(out_dir, "step_reward_speed.png")
    plt.savefig(out1, dpi=150)

    # 2) exploration std
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(env_step, series.get("debug/std_throttle", []), label="std_throttle")
    ax.plot(env_step, series.get("debug/std_steer", []), label="std_steer")
    ax.plot(env_step, series.get("debug/std_y_ref", []), label="std_y_ref")
    ax.set_title("Action Std (Exploration)")
    ax.legend()
    plt.tight_layout()
    out2 = os.path.join(out_dir, "step_action_std.png")
    plt.savefig(out2, dpi=150)

    # 3) reward components（关键项）
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for k in [
        "debug_reward/r_progress",
        "debug_reward/r_speed",
        "debug_reward/r_obs_clear",
        "debug_reward/r_lane",
        "debug_reward/r_danger",
        "debug_reward/r_collision",
    ]:
        if k in series:
            ax.plot(env_step, series[k], alpha=0.5, label=k)
    ax.set_title("Reward Components (Step)")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out3 = os.path.join(out_dir, "step_reward_components.png")
    plt.savefig(out3, dpi=150)

    return [out1, out2, out3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", default="training_log.json")
    parser.add_argument("--wandb-run", default="wandb/latest-run")
    parser.add_argument("--out-dir", default="result/diagnostics")
    args = parser.parse_args()

    # episode-level
    ep_series = load_training_log(args.training_log)
    ep_path = plot_episode_metrics(ep_series, args.out_dir) if ep_series else None

    # step-level
    wandb_hist = find_wandb_history(args.wandb_run)
    step_series = load_wandb_history(wandb_hist) if wandb_hist else {}
    step_paths = plot_step_metrics(step_series, args.out_dir) if step_series else []

    print("episode plot:", ep_path or "N/A")
    print("step plots:", step_paths or "N/A")


if __name__ == "__main__":
    main()
