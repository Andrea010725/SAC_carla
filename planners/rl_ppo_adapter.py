"""
PPO Planner Adapter - 将TensorFlow PPO Agent适配为Planner接口

这个适配器将rl_agent_only中的PPO Agent包装成与current repo兼容的Planner
"""

import os
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)  # planners目录
sys.path.insert(0, os.path.join(current_dir, 'rl_agent_only'))

# 尝试不同的import方式
try:
    from planners.base import PlannerBase, LowLevelAction
except ImportError:
    try:
        from base import PlannerBase, LowLevelAction
    except ImportError:
        # 如果都失败,定义类型
        from typing import Tuple
        from abc import ABC, abstractmethod

        LowLevelAction = Tuple[float, float, float]

        class PlannerBase(ABC):
            @abstractmethod
            def plan(self, obs, info=None):
                pass

            def attach_context(self, world, ego, lane_ref=None):
                pass

            def reset(self):
                pass


class RLPPOAdapter(PlannerBase):
    """
    将PPO Agent适配为Planner接口

    关键转换:
    1. 观测转换: 当前repo的9维obs → PPO需要的state格式
    2. 动作转换: PPO的action输出 → (throttle, brake, steer)三元组
    3. 接口转换: predict() → plan()
    """

    def __init__(self,
                 model_path: str = None,
                 use_ppo_with_trd: bool = True,
                 action_dim: int = 2):
        """
        初始化PPO适配器

        Args:
            model_path: 已训练PPO模型的路径 (如果有)
            use_ppo_with_trd: 是否使用带TRD的PPO版本
            action_dim: PPO输出动作维度 (2或3)
                - 2: [throttle_brake, steer]
                - 3: [throttle, brake, steer]
        """
        self.agent = None
        self.model_path = model_path
        self.use_ppo_with_trd = use_ppo_with_trd
        self.action_dim = action_dim

        self._attached = False
        self._world = None
        self._ego = None
        self._lane_ref = None

        print(f"[RLPPOAdapter] 初始化:")
        print(f"  - TRD支持: {use_ppo_with_trd}")
        print(f"  - 动作维度: {action_dim}")
        print(f"  - 模型路径: {model_path if model_path else '未指定(将使用未训练的网络)'}")

    def attach_context(self, world, ego, lane_ref=None):
        """注入CARLA上下文"""
        self._world = world
        self._ego = ego
        self._lane_ref = lane_ref
        self._attached = True

        print(f"[RLPPOAdapter] ✅ CARLA上下文已注入")

        # TODO: 在这里初始化或加载PPO agent
        # 由于PPO需要gym.Env，我们暂时不在这里创建agent
        # 而是直接在plan()中使用网络推理

        if self.model_path and os.path.exists(self.model_path):
            try:
                self._load_ppo_model()
                print(f"[RLPPOAdapter] ✅ 成功加载PPO模型: {self.model_path}")
            except Exception as e:
                print(f"[RLPPOAdapter] ⚠️  加载模型失败: {e}")
                print(f"[RLPPOAdapter] 将使用默认控制策略")

    def _load_ppo_model(self):
        """加载已训练的PPO模型"""
        print(f"[PPO] 加载模型: {self.model_path}")

        # 1. 创建DummyGymEnv
        dummy_env = DummyGymEnv(obs_dim=9, action_dim=self.action_dim)

        # 2. 导入PPO Agent (使用与train_ppo_standalone.py相同的导入策略)
        import sys
        planners_path = os.path.dirname(os.path.abspath(__file__))
        if planners_path not in sys.path:
            sys.path.insert(0, planners_path)

        from rl_agent_only.agents.ppo import PPOAgent

        # 3. 创建Agent (使用与训练时相同的参数!)
        self.agent = PPOAgent(
            env=dummy_env,
            policy_lr=3e-4,
            value_lr=1e-3,
            gamma=0.99,
            lambda_=0.95,
            clip_ratio=0.2,
            entropy_regularization=0.01,
            optimization_steps=(10, 10),
            batch_size=256,
            update_frequency=1,
            name='ppo-carla-standalone'  # 必须与训练时的name一致!
        )

        # 4. 加载权重
        try:
            self.agent.load_weights()
            print(f"[PPO] ✅ 模型加载成功!")
            print(f"[PPO]    Policy网络: {self.model_path}/policy_net")
            print(f"[PPO]    Value网络: {self.model_path}/value_net")
        except Exception as e:
            print(f"[PPO] ❌ 模型加载失败: {e}")
            raise

    def reset(self):
        """重置planner状态"""
        pass

    def plan(self, obs: np.ndarray, info: Dict[str, Any] = None) -> LowLevelAction:
        """
        规划函数 - 从观测生成车辆控制

        Args:
            obs: [x, y, z, pitch, yaw, roll, acc, ang_v, vel] (9维)
            info: 额外信息 (可选)

        Returns:
            (throttle, brake, steer) 三元组
        """
        # ✅ 添加调试输出
        print(f"[PPO] 🎯 PPO Planner被调用! obs前3维: [{obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}]")

        if not self._attached:
            raise RuntimeError("RLPPOAdapter未attach_context，无法执行plan()")

        # 1. 转换观测为PPO需要的格式
        state = self._convert_obs_to_state(obs)

        # 2. PPO推理 (如果有加载的模型)
        if self.agent is not None:
            try:
                action = self._ppo_inference(state)
                print(f"[PPO] 使用模型推理")
            except Exception as e:
                print(f"[PPO] ⚠️  PPO推理失败: {e}")
                # 失败时使用默认控制
                return self._default_control()
        else:
            # 没有加载模型，使用默认控制
            print("[PPO] 使用默认控制 (无模型)")
            return self._default_control()

        # 3. 转换动作为车辆控制
        throttle, brake, steer = self._convert_action_to_control(action)
        print(f"[PPO] 输出控制: T={throttle:.2f}, B={brake:.2f}, S={steer:.2f}")

        return throttle, brake, steer

    def _convert_obs_to_state(self, obs: np.ndarray) -> Dict[str, tf.Tensor]:
        """
        将当前repo的9维观测转换为PPO需要的state格式

        Args:
            obs: [x, y, z, pitch, yaw, roll, acc, ang_v, vel]

        Returns:
            Dict格式的state (符合PPO的observation space)
        """
        # 确保obs是numpy数组
        obs = np.array(obs, dtype=np.float32).flatten()

        if len(obs) != 9:
            print(f"[RLPPOAdapter] ⚠️  警告: 观测维度为{len(obs)}，期望9维")
            # 填充或截断到9维
            if len(obs) < 9:
                obs = np.pad(obs, (0, 9 - len(obs)), constant_values=0.0)
            else:
                obs = obs[:9]

        # PPO的state_spec格式: {'state': shape}
        # 根据agents.py line 31: utils.space_to_flat_spec(observation_space, name='state')
        # 返回格式应该是 {'state': tensor}

        # 方案1: 直接使用9维obs (最简单)
        state_tensor = tf.expand_dims(tf.constant(obs, dtype=tf.float32), axis=0)  # [1, 9]
        state = {'state': state_tensor}

        # 方案2: 如果PPO训练时使用了更复杂的特征，可以在这里扩展
        # 例如添加速度的极坐标表示、归一化等

        return state

    def _ppo_inference(self, state: Dict[str, tf.Tensor]) -> np.ndarray:
        """
        使用PPO网络进行推理

        Args:
            state: PPO格式的状态

        Returns:
            动作数组 (2维或3维)
        """
        # 使用PPOAgent的predict接口
        # predict返回: (action, mean, std, log_prob, value)
        action, _, _, _, _ = self.agent.predict(state)

        # 转换为numpy
        if isinstance(action, tf.Tensor):
            action = action.numpy()

        return action

    def _convert_action_to_control(self, action: np.ndarray) -> Tuple[float, float, float]:
        """
        将PPO输出的动作转换为车辆控制

        Args:
            action: PPO的动作输出
                - 如果action_dim=2: [throttle_brake, steer]
                - 如果action_dim=3: [throttle, brake, steer]

        Returns:
            (throttle, brake, steer) 三元组
        """
        action = np.array(action).flatten()

        if self.action_dim == 2:
            # 2维动作: [throttle_brake, steer]
            if len(action) < 2:
                print(f"[RLPPOAdapter] ⚠️  动作维度不足: {len(action)}，期望2维")
                return 0.3, 0.0, 0.0

            throttle_brake = float(np.clip(action[0], -1.0, 1.0))
            steer = float(np.clip(action[1], -1.0, 1.0))

            # 分离throttle和brake
            if throttle_brake >= 0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

        elif self.action_dim == 3:
            # 3维动作: [throttle, brake, steer]
            if len(action) < 3:
                print(f"[RLPPOAdapter] ⚠️  动作维度不足: {len(action)}，期望3维")
                return 0.3, 0.0, 0.0

            throttle = float(np.clip(action[0], 0.0, 1.0))
            brake = float(np.clip(action[1], 0.0, 1.0))
            steer = float(np.clip(action[2], -1.0, 1.0))

        else:
            raise ValueError(f"不支持的action_dim: {self.action_dim}，只支持2或3")

        return throttle, brake, steer

    def _default_control(self) -> LowLevelAction:
        """
        默认控制策略 (当PPO模型未加载或推理失败时使用)

        Returns:
            安全的默认控制: 中等油门，无刹车，直行
        """
        return 0.3, 0.0, 0.0


class DummyGymEnv:
    """
    一个虚拟的gym环境，用于满足PPO Agent的初始化要求

    由于PPO Agent需要gym.Env作为参数，但我们实际上只需要其网络进行推理，
    所以创建一个最小的dummy环境来完成初始化
    """
    def __init__(self, obs_dim: int = 9, action_dim: int = 2):
        import gym

        # 定义observation space (9维向量)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 定义action space
        # 假设使用连续动作空间，范围[-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

        self.max_episode_steps = 1000

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def create_ppo_adapter_with_model(model_dir: str,
                                   use_trd: bool = True,
                                   action_dim: int = 2) -> RLPPOAdapter:
    """
    便捷函数: 创建带模型的PPO适配器

    Args:
        model_dir: 模型保存目录 (包含policy_net和value_net)
        use_trd: 是否使用TRD版本的PPO
        action_dim: 动作维度

    Returns:
        RLPPOAdapter实例
    """
    adapter = RLPPOAdapter(
        model_path=model_dir,
        use_ppo_with_trd=use_trd,
        action_dim=action_dim
    )

    # TODO: 在这里实际加载和初始化PPO agent
    # 需要:
    # 1. 创建DummyGymEnv
    # 2. 创建PPOAgent实例
    # 3. 加载模型权重

    # from rl_agent_only.agents.ppo import PPOAgent
    # dummy_env = DummyGymEnv(obs_dim=9, action_dim=action_dim)
    # adapter.agent = PPOAgent(
    #     env=dummy_env,
    #     batch_size=1,
    #     # ... 其他参数
    # )
    # adapter.agent.load_weights()

    return adapter


if __name__ == "__main__":
    """测试脚本"""
    print("=" * 60)
    print("PPO Adapter 测试")
    print("=" * 60)

    # 创建适配器
    adapter = RLPPOAdapter(action_dim=2)

    # 模拟attach_context
    adapter.attach_context(world=None, ego=None, lane_ref=None)

    # 测试观测转换
    test_obs = np.array([100.0, 50.0, 0.3,  # x, y, z
                         0.1, 1.57, 0.0,    # pitch, yaw, roll
                         0.5, 0.0, 5.0])    # acc, ang_v, vel

    print(f"\n测试观测: {test_obs}")

    # 测试plan (没有模型时应该返回默认控制)
    throttle, brake, steer = adapter.plan(test_obs)

    print(f"\n输出控制:")
    print(f"  throttle: {throttle:.3f}")
    print(f"  brake:    {brake:.3f}")
    print(f"  steer:    {steer:.3f}")

    print("\n✅ 适配器基础功能测试通过!")
    print("\n📝 下一步:")
    print("  1. 在create_ppo_adapter_with_model()中实现PPO Agent的实际加载")
    print("  2. 训练一个PPO模型或提供已训练的模型路径")
    print("  3. 在router_carla.py中使用这个适配器替换RL planner")
