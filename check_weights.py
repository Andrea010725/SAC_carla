#!/usr/bin/env python3
"""快速检查PPO权重文件是否存在"""

import os

weights_dir = './weights/ppo-carla-standalone'

print("=" * 70)
print("PPO 权重文件检查")
print("=" * 70)

if not os.path.exists(weights_dir):
    print(f"\n❌ 权重目录不存在: {weights_dir}")
    print("   需要从头开始训练")
    exit(1)

print(f"\n✅ 权重目录存在: {weights_dir}")

# 检查必需的文件
required_files = [
    'policy_net.index',
    'policy_net.data-00000-of-00001',
    'value_net.index',
    'value_net.data-00000-of-00001',
    'checkpoint',
    'config.json'
]

print("\n文件检查:")
all_exist = True
for filename in required_files:
    filepath = os.path.join(weights_dir, filename)
    exists = os.path.exists(filepath)

    if exists:
        size = os.path.getsize(filepath)
        mtime = os.path.getmtime(filepath)
        import datetime
        time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  ✅ {filename:<40} ({size:>8} bytes, {time_str})")
    else:
        print(f"  ❌ {filename:<40} (缺失)")
        all_exist = False

print("\n" + "=" * 70)
if all_exist:
    print("结论: ✅ 所有权重文件完整，可以继续训练")
    print("\n运行 'python train_ppo_standalone.py' 将自动加载这些权重")
else:
    print("结论: ⚠️  部分文件缺失，将从头开始训练")
print("=" * 70)
