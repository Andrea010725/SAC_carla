"""
PPO训练 - 4场景随机切换版本
场景池：ConesScenario, JaywalkerScenario, TrimmaScenario, ConstructionLaneChangeScenario

修改说明：
1. 场景池改为4个指定场景
2. 添加场景触发逻辑支持
3. Wandb配置更新
"""
import sys
import os
import time

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 添加当前项目路径
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# 添加 planners 目录
planners_path = os.path.join(script_dir, "planners")
if planners_path not in sys.path:
    sys.path.insert(0, planners_path)

# 添加 CARLA PythonAPI 路径（用于导入agents模块和carla模块）
carla_api_path = "/home/ajifang/carla/PythonAPI/carla/"
carla_egg_path = "/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
if carla_api_path not in sys.path:
    sys.path.insert(0, carla_api_path)
if carla_egg_path not in sys.path:
    sys.path.insert(0, carla_egg_path)

import gym
import numpy as np

from config import Config
from carla_base.carla_env import CarlaEnv
from training_logger import TrainingLogger

# Wandb导入（优雅处理）
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✅ Wandb已导入")
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Wandb未安装，将跳过在线监控")
    print("   安装命令: pip install wandb")


class CarlaGymEnv(gym.Env):
    """
    CARLA Gym wrapper for PPO - 支持4场景随机切换

    新增功能：
    - 支持场景触发逻辑（JaywalkerScenario）
    - 支持场景更新逻辑（动态场景）
    """

    def __init__(self, config, logger=None, wandb_run=None):
        super().__init__()
        self.config = config
        self.carla_env = CarlaEnv(config, 2000, 8000)
        self.logger = logger
        self.wandb_run = wandb_run

        # ✅ wrapper 直接复用底层 env 的 space
        self.observation_space = self.carla_env.observation_space
        self.action_space = self.carla_env.action_space

        self.max_episode_steps = int(getattr(config, "max_episode_steps", 512))

        # ===== ✅ y_ref 开关交给 CarlaEnv =====
        self.carla_env.use_yref_in_steer = bool(getattr(config, "use_yref_mapping", True))
        self.carla_env.yref_steer_gain = float(getattr(config, "yref_gain", 0.03))

        # y_ref 惩罚
        self.yref_penalty = float(getattr(config, "yref_penalty", 0.0))

        # 饱和率监控
        self.log_steer_saturation = bool(getattr(config, "log_steer_saturation", True))
        self.steer_sat_threshold = float(getattr(config, "steer_sat_threshold", 0.999))

        # ===== 可选：action bias / forced throttle =====
        self.current_episode = 0
        self.use_action_bias = bool(getattr(config, "use_action_bias", False))
        self.use_forced_throttle = bool(getattr(config, "use_forced_throttle", False))
        self.action_bias_strength = float(getattr(config, "action_bias_strength", 0.7))
        self.action_bias_decay = float(getattr(config, "action_bias_decay", 0.999))

        self.episode_collision = False
        self.global_env_step = 0
        self.current_episode = 0
        self._step_count = 0
        self.step_reward_components = {}

        self.wandb_step_log_interval = int(getattr(config, "wandb_step_log_interval", 10))

        print("\n[DEBUG] CarlaEnv y_ref switch after wrapper init:")
        print("  carla_env.use_yref_in_steer =", getattr(self.carla_env, "use_yref_in_steer", None))
        print("  carla_env.yref_steer_gain   =", getattr(self.carla_env, "yref_steer_gain", None))

    def reset(self):
        obs = self.carla_env.reset()
        self.current_episode += 1
        self._step_count = 0
        self.episode_collision = False
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        step_id = int(self.global_env_step)

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size < 3:
            raise ValueError(f"action must have 3 dims [a0,a1,a2], got shape={action.shape}")

        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"⚠️ NaN/Inf action: {action}, fallback to [0.3,0.0,0.0]")
            action = np.array([0.3, 0.0, 0.0], dtype=np.float32)

        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        a0 = float(action[0])
        a1 = float(action[1])
        a2 = float(action[2])

        # ----- bias / forced throttle -----
        applied_bias = 0.0
        forced_throttle = False
        a0_applied = a0

        if self.use_action_bias and self.current_episode < 500:
            applied_bias = self.action_bias_strength * (self.action_bias_decay ** self.current_episode)
            a0_applied = float(np.clip(a0_applied + applied_bias, -1.0, 1.0))

        if self.use_forced_throttle and (-0.1 <= a0_applied <= 0.1):
            a0_applied = 0.2
            forced_throttle = True

        action3 = np.array([a0_applied, a1, a2], dtype=np.float32)
        obs, reward, done, info = self.carla_env.step(action3)
        if info is None:
            info = {}

        # ===== ✅ 场景触发逻辑（新增）=====
        if hasattr(self.carla_env, 'scenario_instance') and self.carla_env.scenario_instance:
            scenario = self.carla_env.scenario_instance

            # JaywalkerScenario: 检查并触发行人横穿
            if hasattr(scenario, 'check_and_trigger'):
                try:
                    ego_loc = self.carla_env.ego.get_location()
                    scenario.check_and_trigger(ego_loc)
                except Exception as e:
                    print(f"⚠️ check_and_trigger failed: {e}")

            # JaywalkerScenario: 更新行人移动
            if hasattr(scenario, 'tick_update'):
                try:
                    scenario.tick_update()
                except Exception as e:
                    print(f"⚠️ tick_update failed: {e}")

        # ----- y_ref penalty -----
        r_yref = 0.0
        if self.yref_penalty > 0.0:
            r_yref = -float(self.yref_penalty * abs(a2))
            reward = float(reward) + r_yref

        # ----- reward components -----
        comps = {}
        if isinstance(getattr(self.carla_env, "last_reward_components", None), dict):
            comps = self.carla_env.last_reward_components.copy()
        comps["r_yref"] = float(r_yref)
        self.step_reward_components = comps

        # ----- info -----
        info.update({
            "wrapped_a0_raw": float(a0),
            "wrapped_a0_applied": float(a0_applied),
            "wrapped_applied_bias": float(applied_bias),
            "wrapped_forced_throttle": int(forced_throttle),
            "wrapper_yref_penalty": float(self.yref_penalty),
            "env_step": float(step_id),
        })

        # ----- logger -----
        if self.logger is not None:
            speed = float(info.get("speed", 0.0))
            collision = bool(info.get("collision", 0.0))
            if collision:
                self.episode_collision = True
            self.logger.log_step(speed=speed, collision=collision)

        # ----- W&B log -----
        if self.wandb_run is not None and (step_id % max(1, self.wandb_step_log_interval) == 0):
            speed = float(info.get("speed", 0.0))
            steer_raw = float(info.get("raw_steer", a1))
            y_ref = float(info.get("raw_y_ref", a2))
            steer_applied = float(info.get("applied_steer", steer_raw))

            yref_used = int(info.get("yref_used", 1))
            delta_total = float(steer_applied - steer_raw)
            delta_from_yref = float(info.get("steer_delta_from_yref", 0.0)) if yref_used else 0.0

            steer_saturated = 0
            if self.log_steer_saturation:
                steer_saturated = int(abs(steer_applied) >= self.steer_sat_threshold)

            step_log = {
                "env_step": int(step_id),
                "step/reward": float(reward),
                "step/speed": float(speed),

                "step/action_throttle_brake_raw": float(a0),
                "step/action_throttle_brake_applied": float(a0_applied),
                "step/action_steer_raw": float(steer_raw),
                "step/y_ref": float(y_ref),

                "step/action_steer_applied": float(steer_applied),
                "debug/steer_postprocess_delta": float(delta_total),
                "debug/steer_delta_from_yref": float(delta_from_yref),
                "debug/steer_saturated": int(steer_saturated),

                "step/applied_bias": float(applied_bias),
                "step/forced_throttle": int(forced_throttle),

                "debug/yref_used": float(yref_used),
                "debug/yref_steer_gain": float(info.get("yref_steer_gain",
                                                        getattr(self.carla_env, "yref_steer_gain", 0.0))),
            }

            sum_components = float(sum(self.step_reward_components.values())) if self.step_reward_components else 0.0
            total_reward = float(reward)
            step_log.update({
                "debug_reward/sum_components": sum_components,
                "debug_reward/total_reward": total_reward,
                "debug_reward/components_diff": float(total_reward - sum_components),
            })

            for k, v in self.step_reward_components.items():
                step_log[f"debug_reward/{k}"] = float(v)

            self.wandb_run.log(step_log, step=int(step_id), commit=True)

        self._step_count += 1
        self.global_env_step += 1
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def close(self):
        if hasattr(self, "carla_env") and self.carla_env is not None:
            self.carla_env.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, mode="human"):
        pass


def train_with_logging(agent, env, logger, wandb_run=None, episodes=300, timesteps=512, save_every=100):
    """
    带日志记录和Wandb监控的PPO训练循环 + ✅自动早停/收敛判定
    """
    import time
    import os
    import numpy as np
    from collections import deque
    from planners.rl_agent_only import utils

    assert episodes % agent.update_frequency == 0
    agent.memory = agent.get_memory()

    policy_loss_tracker = 0.0
    value_loss_tracker = 0.0
    entropy_tracker = 0.0
    update_count = 0

    # ============================================================
    # ✅ Early Stop / Convergence 规则
    # ============================================================
    W = 20  # 滑动窗口长度
    MIN_EP = 30  # 至少跑到这个 episode 才开始早停判断

    # ---- 收敛判据（满足就停）----
    TARGET_SUCCESS = 0.80       # success率 >= 0.80
    TARGET_RAN_FULL = 0.80      # ran_full率 >= 0.80
    MAX_COLLISION = 0.10        # collision率 <= 0.10
    NEED_STABLE_WINDOWS = 3     # 连续多少个窗口满足才算"稳定收敛"

    # ---- 平台期判据（长时间没提升就停）----
    PLATEAU_PATIENCE = 8        # 连续多少个窗口没提升就停
    IMPROVE_EPS = 0.01          # success_rate 提升小于该值视为"没提升"

    # ---- 崩溃判据（明显坏掉就停）----
    BAD_PATIENCE = 3            # 连续 BAD_PATIENCE 个窗口崩 → 停
    BAD_COLLISION = 0.80        # collision率 > 0.80 认为崩
    BAD_RAN_FULL = 0.10         # ran_full率 < 0.10 认为崩

    # 统计窗口
    dq_success = deque(maxlen=W)
    dq_ran_full = deque(maxlen=W)
    dq_no_collision = deque(maxlen=W)
    dq_avg_rps = deque(maxlen=W)

    stable_good_count = 0
    bad_count = 0

    best_succ_rate = -1.0
    best_succ_episode = 0
    plateau_count = 0

    def _mean(dq):
        return float(sum(dq) / max(1, len(dq)))

    for episode in range(1, episodes + 1):
        done_reason = "unknown"
        last_info = {}

        episode_entropy_sum = 0.0
        episode_entropy_count = 0

        logger.start_episode(episode)

        agent.seed_regularization()
        agent.on_episode_start()
        agent.reset()

        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        t0 = time.time()

        episode_speed_sum = 0.0
        episode_collision_count = 0
        episode_reward_components = {}
        episode_duration = 0.0

        # ---------------------- rollout ----------------------
        for t in range(1, timesteps + 1):
            if isinstance(state, dict):
                state = {f"state_{k}": v for k, v in state.items()}
            state_t = utils.to_tensor(state)

            action, mean, std, log_prob, value = agent.predict(state_t)

            # ===== entropy from std (Gaussian) =====
            std_np = std.numpy() if hasattr(std, "numpy") else np.asarray(std)
            std_np = np.asarray(std_np).reshape(-1)
            var = np.square(std_np) + 1e-8
            ent = 0.5 * np.sum(np.log(2.0 * np.pi * np.e * var))
            episode_entropy_sum += float(ent)
            episode_entropy_count += 1

            # ===== W&B debug std =====
            if wandb_run is not None:
                step_id = int(env.global_env_step)
                if (step_id % max(1, env.wandb_step_log_interval)) == 0:
                    std_np2 = std.numpy() if hasattr(std, "numpy") else np.asarray(std)
                    std_np2 = np.asarray(std_np2).reshape(-1)
                    std_th = float(std_np2[0]) if std_np2.size > 0 else 0.0
                    std_st = float(std_np2[1]) if std_np2.size > 1 else 0.0
                    std_yr = float(std_np2[2]) if std_np2.size > 2 else 0.0
                    std_lon_mean = std_th
                    std_lat_mean = float(np.mean([std_st, std_yr]))

                    wandb_run.log(
                        {
                            "env_step": int(step_id),
                            "debug/std_lon_mean": std_lon_mean,
                            "debug/std_lat_mean": std_lat_mean,
                            "debug/std_throttle": std_th,
                            "debug/std_steer": std_st,
                            "debug/std_y_ref": std_yr,
                        },
                        step=step_id,
                        commit=False,
                    )

            action_env = agent.convert_action(action)
            next_state, reward, done, info = env.step(action_env)
            last_info = info or {}

            episode_reward += float(reward)
            episode_steps = t

            # ✅ 累积 reward 分解
            if (not episode_reward_components) and getattr(env, "step_reward_components", None):
                episode_reward_components = {k: 0.0 for k in env.step_reward_components.keys()}

            for k in episode_reward_components.keys():
                episode_reward_components[k] += float(env.step_reward_components.get(k, 0.0))

            # 速度统计
            if isinstance(next_state, (list, tuple, np.ndarray)) and len(next_state) > 8:
                episode_speed_sum += float(next_state[8])

            # 碰撞统计
            if last_info.get("collision", 0.0):
                episode_collision_count += 1

            # 存经验
            agent.log(actions=action, action_env=action_env, rewards=reward,
                      distribution_mean=mean, distribution_std=std)
            agent.memory.append(state_t, action, reward, value, log_prob)

            state = next_state

            if done or (t == timesteps):
                episode_duration = time.time() - t0
                done_reason = str(last_info.get("done_reason", "unknown"))

                print(
                    f"Episode {episode} terminated after {t} timesteps in {round(episode_duration, 3)}s "
                    f"with reward {round(episode_reward, 3)}."
                )
                agent.log(timestep=t)

                if isinstance(state, dict):
                    state = {f"state_{k}": v for k, v in state.items()}
                state_tt = utils.to_tensor(state)

                last_value = agent.network.predict_last_value(
                    state_tt, timestep=(t + 1) / timesteps, is_terminal=done
                )
                agent.end_episode(last_value, append=agent.update_frequency > 1)
                break

        # ---------------------- episode stats ----------------------
        avg_speed = episode_speed_sum / episode_steps if episode_steps > 0 else 0.0
        logger.log_episode_reward(episode_reward)

        # ---------------------- update ----------------------
        if episode % agent.update_frequency == 0:
            temp_policy_loss, temp_value_loss, temp_entropy = [], [], []

            value_batches = agent.get_value_batches()
            policy_batches = agent.get_policy_batches()

            for _ in range(agent.optimization_steps["policy"]):
                for data_batch in policy_batches:
                    agent.seed_regularization()
                    total_loss, policy_grads = agent.get_policy_gradients(data_batch)

                    policy_loss_val = float(total_loss[0].numpy()) if isinstance(total_loss, tuple) else float(total_loss.numpy())
                    if not (np.isnan(policy_loss_val) or np.isinf(policy_loss_val)):
                        temp_policy_loss.append(policy_loss_val)

                    if hasattr(agent, "_logs") and "entropy" in agent._logs:
                        entropy_val = agent._logs["entropy"]
                        if isinstance(entropy_val, (list, tuple)):
                            entropy_val = entropy_val[-1] if entropy_val else 0.0
                        if not (np.isnan(entropy_val) or np.isinf(entropy_val)):
                            temp_entropy.append(float(entropy_val))

                    agent.update_policy(policy_grads)

            for _ in range(agent.optimization_steps["value"]):
                for data_batch in value_batches:
                    agent.seed_regularization()
                    value_loss, value_grads = agent.get_value_gradients(data_batch)
                    value_loss_val = float(value_loss.numpy())
                    if not (np.isnan(value_loss_val) or np.isinf(value_loss_val)):
                        temp_value_loss.append(value_loss_val)
                    agent.update_value(value_grads)

            if temp_policy_loss:
                policy_loss_tracker = sum(temp_policy_loss) / len(temp_policy_loss)
            if temp_value_loss:
                value_loss_tracker = sum(temp_value_loss) / len(temp_value_loss)
            if temp_entropy:
                entropy_tracker = sum(temp_entropy) / len(temp_entropy)

            agent.memory.delete()
            agent.memory = agent.get_memory()
            update_count += 1

        elif agent.update_frequency > 1:
            agent.memory.rewards = agent.memory.rewards[:-1]
            agent.memory.values = agent.memory.values[:-1]

        logger.log_training_metrics(
            policy_loss=policy_loss_tracker,
            value_loss=value_loss_tracker,
            entropy=entropy_tracker
        )
        logger.end_episode()

        # ---------------------- success / ran_full / no_collision ----------------------
        timeout_flag = bool(float(last_info.get("timeout", 0.0)) > 0.5)
        ran_full = bool(timeout_flag or (episode_steps >= timesteps))

        bad_reasons = {"collision", "offroad", "no_progress"}
        no_collision = bool(episode_collision_count == 0)
        success = int(ran_full and no_collision and (done_reason not in bad_reasons))

        # ============================================================
        # ✅ Early Stop 统计
        # ============================================================
        dq_success.append(int(success))
        dq_ran_full.append(int(ran_full))
        dq_no_collision.append(int(no_collision))
        dq_avg_rps.append(float(episode_reward / max(1, episode_steps)))

        succ_rate = _mean(dq_success)
        ran_full_rate = _mean(dq_ran_full)
        no_collision_rate = _mean(dq_no_collision)
        collision_rate = 1.0 - no_collision_rate

        # ---- 收敛判据 ----
        if (episode >= MIN_EP and len(dq_success) == W and
            succ_rate >= TARGET_SUCCESS and
            ran_full_rate >= TARGET_RAN_FULL and
            collision_rate <= MAX_COLLISION):
            stable_good_count += 1
        else:
            stable_good_count = 0

        # ---- 崩溃判据 ----
        if (episode >= MIN_EP and len(dq_success) == W and
            (collision_rate > BAD_COLLISION or ran_full_rate < BAD_RAN_FULL)):
            bad_count += 1
        else:
            bad_count = 0

        # ---- 平台期判据 ----
        if episode >= MIN_EP and len(dq_success) == W:
            if succ_rate > best_succ_rate + IMPROVE_EPS:
                best_succ_rate = succ_rate
                best_succ_episode = episode
                plateau_count = 0
            else:
                plateau_count += 1

        # ---------------------- W&B episode log ----------------------
        if wandb_run is not None:
            avg_reward_components = {
                f"reward/{k}": (v / episode_steps if episode_steps > 0 else 0.0)
                for k, v in episode_reward_components.items()
            }

            wandb_metrics = {
                "episode_num": int(episode),
                "episode/ran_full": int(ran_full),
                "episode/no_collision": int(no_collision),
                "episode/success": int(success),
                "episode/done_reason": done_reason,

                "episode/total_reward": float(episode_reward),
                "episode/avg_reward_per_step": float(episode_reward / max(1, episode_steps)),
                "episode/steps": int(episode_steps),
                "episode/duration": float(episode_duration),
                "episode/avg_speed": float(avg_speed),
                "episode/collision_count": int(episode_collision_count),

                "training/policy_loss": float(policy_loss_tracker),
                "training/value_loss": float(value_loss_tracker),
                "training/entropy": float(entropy_tracker),
                "training/update_count": int(update_count),
                "training/entropy_rollout_mean": float(episode_entropy_sum / max(1, episode_entropy_count)),

                # ✅ early-stop 观测指标
                "early/success_rate_W": float(succ_rate),
                "early/ran_full_rate_W": float(ran_full_rate),
                "early/collision_rate_W": float(collision_rate),
                "early/stable_good_count": int(stable_good_count),
                "early/bad_count": int(bad_count),
                "early/plateau_count": int(plateau_count),
                "early/best_succ_rate": float(best_succ_rate),
                "early/best_succ_episode": int(best_succ_episode),
            }
            wandb_metrics.update(avg_reward_components)

            last_step = int(env.global_env_step) - 1
            wandb_metrics["env_step"] = last_step
            wandb_run.log(wandb_metrics, step=last_step, commit=True)

        # ---------------------- summaries / record ----------------------
        agent.log(episode_rewards=episode_reward)
        agent.write_summaries()

        if agent.should_record:
            agent.record(episode)

        agent.on_episode_end()

        # ---------------------- periodic save ----------------------
        if episode % save_every == 0:
            agent.save()
            if wandb_run is not None:
                checkpoint_path = os.path.join(agent.base_path, "policy_net.index")
                if os.path.exists(checkpoint_path):
                    wandb_run.save(os.path.join(agent.base_path, "policy_net*"))
                    wandb_run.save(os.path.join(agent.base_path, "value_net*"))

        if episode % 10 == 0:
            logger.print_summary(window=10)

        # ============================================================
        # ✅ Early Stop 触发
        # ============================================================
        if episode >= MIN_EP and len(dq_success) == W:
            if stable_good_count >= NEED_STABLE_WINDOWS:
                print(
                    f"\n✅ EarlyStop: CONVERGED. "
                    f"succ_rate(W)={succ_rate:.3f}, ran_full_rate(W)={ran_full_rate:.3f}, "
                    f"collision_rate(W)={collision_rate:.3f}\n"
                )
                agent.save()
                break

            if bad_count >= BAD_PATIENCE:
                print(
                    f"\n⚠️ EarlyStop: DIVERGED/BAD. "
                    f"succ_rate(W)={succ_rate:.3f}, ran_full_rate(W)={ran_full_rate:.3f}, "
                    f"collision_rate(W)={collision_rate:.3f}\n"
                )
                agent.save()
                break

            if plateau_count >= PLATEAU_PATIENCE:
                print(
                    f"\n⏸️ EarlyStop: PLATEAU. "
                    f"best_succ_rate={best_succ_rate:.3f} at ep {best_succ_episode}, "
                    f"current succ_rate(W)={succ_rate:.3f}\n"
                )
                agent.save()
                break


def train_ppo():
    """统一走 train_ppo_with_wandb 的稳定训练链（4scenarios 档）"""
    from train_ppo_with_wandb import train_ppo as train_ppo_unified

    print("[Route] train_ppo_4scenarios -> train_ppo_with_wandb(profile='4scenarios')")
    return train_ppo_unified(profile="4scenarios")


if __name__ == "__main__":
    model_path = train_ppo()
    print(f"\n下一步: 在router中使用训练好的模型")
    print(f"  ppo_model_path='{model_path}'")
