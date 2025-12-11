"""测试Action Bias是否正常工作"""
import numpy as np

# 模拟CarlaGymEnv的action bias逻辑
class TestEnv:
    def __init__(self):
        self.current_episode = 0
        self.action_bias_strength = 0.3
        self.action_bias_decay = 0.995
        self._step_count = 0

    def reset(self):
        self.current_episode += 1
        self._step_count = 0
        print(f"\n=== Episode {self.current_episode} ===")

    def step(self, action):
        if self.current_episode < 100:
            bias = self.action_bias_strength * (self.action_bias_decay ** self.current_episode)
            action = np.array(action, dtype=np.float32)
            original_action = action[0]
            action[0] = float(np.clip(action[0] + bias, -1.0, 1.0))

            if self._step_count < 5:
                print(f"Step {self._step_count}: "
                      f"original={original_action:.3f}, "
                      f"bias={bias:.3f}, "
                      f"final={action[0]:.3f}")
            self._step_count += 1

        return action

# 测试
env = TestEnv()

print("=" * 60)
print("测试Action Bias")
print("=" * 60)

# Episode 1
env.reset()
for i in range(5):
    action = np.array([0.0, 0.0])  # PPO初期可能输出接近0的action
    final_action = env.step(action)

# Episode 10
for _ in range(9):
    env.reset()

env.reset()
for i in range(5):
    action = np.array([0.0, 0.0])
    final_action = env.step(action)

# Episode 50
for _ in range(40):
    env.reset()

env.reset()
for i in range(5):
    action = np.array([0.0, 0.0])
    final_action = env.step(action)

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n预期结果:")
print("- Episode 1: bias ≈ 0.299, final ≈ 0.299")
print("- Episode 10: bias ≈ 0.285, final ≈ 0.285")
print("- Episode 50: bias ≈ 0.233, final ≈ 0.233")
print("\n如果final都是0，说明action bias没有生效！")
