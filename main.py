
import sys
sys.path.append("/home/ajifang/carla/PythonAPI/carla/")
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")


import os
import time
import numpy as np
import logging
from tensorboardX import SummaryWriter
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

import carla
# import carla_base

from carla_base.carla_env import CarlaEnv
from replay_memory import ReplayMemory
from env_wrapper import ParallelEnv, LocalEnv
from agent_base import SAC_Model, SAC_Alg, SAC_Agent  # Choose base wrt which deep-learning framework you are using
from config import Config
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


class TrainPipeline(object):
    def __init__(self,config):
        self.config = config

        # Parallel environments for training
        self.parallel_envs = ParallelEnv(self.config)

        # env for eval
        # self.eval_env = LocalEnv(self.config)

        # Initialize model, algorithm, agent
        self.model     = SAC_Model(self.config.obs_dim, self.config.action_dim)
        self.algorithm = SAC_Alg( self.model, self.config)
        self.agent     = SAC_Agent(self.algorithm, self.config)

        # Initialize replay_memory
        self.men = ReplayMemory(self.config)

        # tensorboard record
        record_summarywriter_path = os.path.join(self.config.record_path,"runs")
        self.writer = SummaryWriter(record_summarywriter_path)

        # logging record
        os.makedirs(os.path.join(self.config.record_path, "output_logger/"), exist_ok=True)
        logging.basicConfig(filename = os.path.join(self.config.record_path, "output_logger/")+"/sac_obsType_"
                                            + config.observations_type +"_train"+".log", level = logging.INFO)
        logging.info("-----------------Carla_SAC-------------------")


    def train(self):
        total_steps = 0
        last_save_steps = 0
        test_flag = 0
        avg_reward = 0

        obs_list = self.parallel_envs.reset()

        print("total_steps: ",total_steps)
        while total_steps < self.config.train_total_steps:
            # Train episode
            if self.men.size() < self.config.warmup_steps:   # 刚开始时先进行一下预热，往经验池填充一些随即样本
            #     action_list = [
            #         np.random.uniform(-1, 1, size=self.config.action_dim)
            #         for _ in range(self.config.num_parallel_envs)
            #     ]
            # else:
            #     action_list = self.agent.sample(obs_list)   # 并行获取动作
                action_list = [
                    np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
                    for _ in range(self.config.num_parallel_envs)
                ]
            else:
                action_list = [
                    np.random.uniform(-1, 1, size=self.config.action_dim).astype(np.float32)
                    for _ in range(self.config.num_parallel_envs)
                ]

            next_obs_list, reward_list, done_list, info_list = self.parallel_envs.step(
                action_list)

            # Store data in replay memory
            for i in range(self.config.num_parallel_envs):
                self.men.append(obs_list[i], action_list[i], reward_list[i],
                        next_obs_list[i], done_list[i])

            obs_list = self.parallel_envs.get_obs()
            total_steps = self.parallel_envs.total_steps

            # 每100步打印一次 planner 统计
            if total_steps % 100 == 0:
                stats = self.parallel_envs.planner_stats
                print(f'[Step {total_steps}] Planner stats - RULE: {stats["RULE"]}, IL: {stats["IL"]}, RL: {stats["RL"]}')

            # Train agent after collecting sufficient data
            if self.men.size() >= self.config.warmup_steps:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = self.men.sample_batch(
                    self.config.batch_size)
                critic_loss, actor_loss = self.agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                            batch_terminal)
                print('\n total_steps: {}, critic_loss: {}, actor_loss: {}'.format(total_steps, critic_loss, actor_loss))
                logging.info('\n total_steps: {}, critic_loss: {}, actor_loss: {}'.format(total_steps, critic_loss, actor_loss))

            # Save agent
            if total_steps > int(self.config.save_interval_steps) and total_steps > last_save_steps + int(1e4):
                self.agent.save(self.config.record_path+'/step_{}_reward_{}.model'.format(total_steps, avg_reward))
                last_save_steps = total_steps

            # # Evaluate episode
            if (total_steps + 1) // self.config.test_interval_steps >= test_flag:
                while (total_steps + 1) // self.config.test_interval_steps >= test_flag:
                    test_flag += 1
                avg_reward = self.run_evaluate_episodes()
                self.writer.add_scalars('episode_reward', {'episode_reward':avg_reward}, total_steps)

                logging.info(
                    '\nTotal steps {}, Evaluation over {} episodes, Average reward: {}'
                    .format(total_steps, self.config.eval_episodes, avg_reward))

                print('\nTotal steps {}, Evaluation over {} episodes, Average reward: {}'.
                    format(total_steps, self.config.eval_episodes, avg_reward))


    # Runs policy for 3 episodes by default and returns average reward
    def run_evaluate_episodes(self):  # 3局的reward之和再做平均
        avg_reward = 0.
        for k in range(self.config.eval_episodes):
            obs = self.parallel_envs.reset()
            done = False
            steps = 0
            while not done and steps < self.config.max_episode_steps:
                steps += 1

                action = self.agent.predict(obs[0])   # obs跟 train里面的obs获取方式不一样，train里面是通过get_obs 函数获取
                obs, reward, done, _ = self.parallel_envs.step(action)
                avg_reward += reward[0]
        avg_reward /= self.config.eval_episodes
        return avg_reward

def env_test(config):
    """
    测试场景和Rule Planner的表现
    """
    import carla
    sys.path.insert(0, "/home/ajifang/SAC_carla/agent_base")
    from rule_based_agent import LaneRef
    from rule_planner_adapter import RulePlannerAdapter

    print("=" * 60)
    print("场景和Rule Planner测试")
    print("=" * 60)
    print(f"场景模式: {config.scenario}")
    print(f"地图: {config.map_name}")
    print(f"最大步数: {config.max_episode_steps}")
    print("=" * 60)

    # 创建环境
    env = CarlaEnv(config, config.test_carla_port, config.test_carla_tm_port)
    print("\n环境信息:")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  动作范围: {env.action_space.low} ~ {env.action_space.high}")

    # 初始化Rule Planner
    print("\n初始化Rule Planner...")
    # 修改目标速度：v_ref_base 单位是 m/s
    # 12 m/s = 43.2 km/h (默认)
    # 15 m/s = 54 km/h (快)
    # 10 m/s = 36 km/h (慢)
    rule_planner = RulePlannerAdapter(v_ref_base=12.0)  # 可修改这里的速度

    # Reset环境
    print("\n重置环境...")
    obs = env.reset()
    print(f"初始观测: {obs}")

    # 构建LaneRef（参考线）
    print("\n构建LaneRef参考线...")
    try:
        world = env.world
        amap = world.get_map()
        ego = env.ego

        ego_loc = ego.get_location()
        ego_wp = amap.get_waypoint(ego_loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)

        if ego_wp is None:
            raise RuntimeError("无法获取自车waypoint")

        # 构建LaneRef
        lane_ref = LaneRef(amap, seed_wp=ego_wp, step=1.0, max_len=500.0)
        print(f"  LaneRef构建成功，采样点数: {len(lane_ref.P)}")

        # Attach context到Rule Planner
        rule_planner.attach_context(world=world, ego=ego, ref=lane_ref)
        print("  Rule Planner上下文注入成功")

    except Exception as e:
        print(f"  [错误] LaneRef构建失败: {e}")
        print("  将使用简单控制策略")
        lane_ref = None

    # 测试循环
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)

    steps = 0
    done = False
    total_reward = 0.0
    collision_count = 0

    # 统计信息
    avg_speed = 0.0
    max_speed = 0.0

    while not done and steps < env.max_episode_steps:
        steps += 1

        # 使用Rule Planner生成控制
        if lane_ref is not None:
            try:
                throttle, brake, steer = rule_planner.plan(obs, info={})
                # 转换为env期望的格式 [throttle_brake, steer]
                throttle_brake = throttle if brake <= 1e-6 else -brake
                action = [throttle_brake, steer]
            except Exception as e:
                print(f"  [步骤{steps}] Rule Planner失败: {e}，使用默认控制")
                action = [0.3, 0.0]  # 小油门直行
        else:
            # 无LaneRef时的简单策略：小油门直行
            action = [0.3, 0.0]

        # 执行动作
        next_obs, reward, done, info = env.step(action)
        total_reward += reward

        # 提取速度信息（观测的第9维是速度）
        if len(obs) >= 9:
            speed = obs[8]  # m/s
            avg_speed += speed
            max_speed = max(max_speed, speed)

        # 检测碰撞
        if hasattr(env, 'collision') and env.collision:
            collision_count += 1
            env.collision = False  # 重置碰撞标志

        # 每50步打印一次状态
        if steps % 50 == 0:
            print(f"  [步骤{steps}] 奖励: {reward:.2f}, 累计: {total_reward:.2f}, "
                  f"速度: {speed:.2f} m/s, 碰撞: {collision_count}")

        obs = next_obs

    # 结果统计
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"总步数: {steps}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {total_reward/steps:.4f}")
    print(f"平均速度: {avg_speed/steps:.2f} m/s")
    print(f"最大速度: {max_speed:.2f} m/s")
    print(f"碰撞次数: {collision_count}")
    print(f"是否完成: {'是' if done else '否（达到最大步数）'}")
    print("=" * 60)

    # 关闭环境
    env.close()



if __name__ == "__main__":

    start = time.time()
    config = Config()

    # ========== 选择运行模式 ==========
    # 模式1: 训练模式
    # training_pipeline = TrainPipeline(config)
    # training_pipeline.train()

    # 模式2: 测试模式（测试场景和Rule Planner）
    env_test(config)

    end = time.time()
    print("run time:%.4fs" % (end - start))


