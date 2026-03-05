#!/usr/bin/env python3
"""
使用 train_ppo_with_wandb.py 训练得到的 PPO 模型，
在 cones 场景下测试并记录逐 step 指标（默认 100 步）。
"""

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
import carla

# 路径准备（与训练脚本保持一致）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PLANNERS_DIR = os.path.join(SCRIPT_DIR, "planners")
if PLANNERS_DIR not in sys.path:
    sys.path.insert(0, PLANNERS_DIR)

CARLA_API_PATH = "/home/ajifang/carla/PythonAPI/carla/"
CARLA_EGG_PATH = "/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
if CARLA_API_PATH not in sys.path:
    sys.path.insert(0, CARLA_API_PATH)
if CARLA_EGG_PATH not in sys.path:
    sys.path.insert(0, CARLA_EGG_PATH)

from config import Config
from train_ppo_with_wandb import CarlaGymEnv
from rl_agent_only import utils
from rl_agent_only.agents.ppo import PPOAgent


CSV_FIELDS = [
    "run_id",
    "method_id",
    "planner_id",
    "scenario",
    "episode_index",
    "seed",
    "step",
    "sim_time_s",
    "collision",
    "speed_mps",
    "throttle",
    "steer",
    "brake",
    "eva_total",
    "eva_safety",
    "eva_comfort",
    "eva_efficiency",
    "dev_m",
    "heading_err_rad",
    "ttc_s",
    "obs_dist_m",
]

FIXED_METHOD_ID = "il planner"
FIXED_PLANNER_ID = "il planner"


@dataclass
class EVAConfig:
    w_saf: float = 1.0
    w_com: float = 0.2
    w_eff: float = 0.1

    w_col: float = 1.0
    w_dev: float = 0.3
    w_head: float = 0.2
    w_off: float = 0.6
    w_obs: float = 0.6
    w_spd_pen: float = 0.2

    w_steer: float = 0.3
    w_jerk: float = 0.4
    w_brake: float = 0.3

    w_spd: float = 0.6
    w_prog: float = 0.4

    max_dev: float = 2.0
    max_head: float = 0.35
    speed_limit: float = 12.0
    min_speed: float = 0.5
    ttc_crit: float = 3.0
    obs_dist_crit: float = 8.0
    brake_thr: float = 0.5


class EvaEvaluator:
    def __init__(self, cfg: EVAConfig = EVAConfig()):
        self.cfg = cfg
        self.world = None
        self.ego = None
        self.prev_speed = None
        self.prev_acc = None
        self.prev_time = None

    def attach(self, world: carla.World, ego: carla.Actor) -> None:
        self.world = world
        self.ego = ego
        self.prev_speed = None
        self.prev_acc = None
        self.prev_time = None

    def _get_lateral_and_heading_error(self) -> Tuple[float, float]:
        if self.world is None or self.ego is None:
            return 0.0, 0.0

        amap = self.world.get_map()
        ego_tf = self.ego.get_transform()
        wp = amap.get_waypoint(
            ego_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            return 0.0, 0.0

        dx = ego_tf.location.x - wp.transform.location.x
        dy = ego_tf.location.y - wp.transform.location.y
        yaw_wp = math.radians(wp.transform.rotation.yaw)
        nx = -math.sin(yaw_wp)
        ny = math.cos(yaw_wp)
        dev = dx * nx + dy * ny

        yaw_ego = math.radians(ego_tf.rotation.yaw)
        head_err = (yaw_ego - yaw_wp + math.pi) % (2.0 * math.pi) - math.pi
        return float(dev), float(head_err)

    def _is_offroad(self) -> bool:
        if self.world is None or self.ego is None:
            return False
        amap = self.world.get_map()
        wp = amap.get_waypoint(
            self.ego.get_location(),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        return wp is None

    def _nearest_obstacle_info(self, extra_candidates=None) -> Tuple[float, float]:
        if self.world is None or self.ego is None:
            return 999.0, 999.0

        ego_tf = self.ego.get_transform()
        ego_vel = self.ego.get_velocity()
        ego_speed = math.hypot(ego_vel.x, ego_vel.y)
        fwd = ego_tf.get_forward_vector()

        nearest_dist = 999.0
        ttc = 999.0

        actors = []
        if extra_candidates is not None:
            actors.extend(list(extra_candidates))
        else:
            vehicles = list(self.world.get_actors().filter("vehicle.*"))
            pedestrians = list(self.world.get_actors().filter("walker.pedestrian.*"))
            static_props = []
            for p in self.world.get_actors().filter("static.prop.*"):
                tid = getattr(p, "type_id", "").lower()
                if ("trafficcone" in tid) or ("cone" in tid) or ("barrier" in tid) or ("construction" in tid):
                    static_props.append(p)
            actors.extend(vehicles + pedestrians + static_props)

        # 去重
        seen_ids = set()
        uniq_actors = []
        for a in actors:
            aid = getattr(a, "id", None)
            if aid is None:
                continue
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            uniq_actors.append(a)

        for actor in uniq_actors:
            if actor.id == self.ego.id:
                continue

            try:
                loc = actor.get_location()
            except Exception:
                continue
            dx = loc.x - ego_tf.location.x
            dy = loc.y - ego_tf.location.y
            ahead = dx * fwd.x + dy * fwd.y
            if ahead <= 0.0:
                continue

            dist = math.hypot(dx, dy)
            if dist < nearest_dist:
                nearest_dist = dist
                rel_v = ego_speed
                try:
                    actor_vel = actor.get_velocity()
                    rel_v = ego_speed - math.hypot(actor_vel.x, actor_vel.y)
                except Exception:
                    pass
                if rel_v > 0.1:
                    ttc = dist / rel_v
                else:
                    ttc = 999.0

        return nearest_dist, ttc

    def _speed_reward(self, speed: float) -> float:
        v_des = min(self.cfg.speed_limit, max(self.cfg.min_speed, self.cfg.speed_limit))
        sigma = 2.0
        return math.exp(-((speed - v_des) ** 2) / (2.0 * sigma ** 2))

    def tick(self, collision: bool = False, extra_candidates=None) -> Dict[str, float]:
        if self.world is None or self.ego is None:
            return {}

        snap = self.world.get_snapshot()
        now = snap.timestamp.elapsed_seconds
        vel = self.ego.get_velocity()
        speed = math.hypot(vel.x, vel.y)

        if self.prev_speed is None or self.prev_time is None:
            acc = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            acc = (speed - self.prev_speed) / dt

        if self.prev_acc is None or self.prev_time is None:
            jerk = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            jerk = (acc - self.prev_acc) / dt

        ctrl = self.ego.get_control()
        steer = float(ctrl.steer)
        dev, head_err = self._get_lateral_and_heading_error()
        offroad = 1.0 if self._is_offroad() else 0.0
        obs_dist, ttc = self._nearest_obstacle_info(extra_candidates=extra_candidates)

        r_col = -1.0 if collision else 0.0
        r_dev = -min(abs(dev) / self.cfg.max_dev, 1.0)
        r_head = -min(abs(head_err) / self.cfg.max_head, 1.0)
        r_off = -offroad
        r_obs = -1.0 if (obs_dist < self.cfg.obs_dist_crit or ttc < self.cfg.ttc_crit) else 0.0
        r_spd_pen = -max(0.0, (speed - self.cfg.speed_limit) / max(1e-3, self.cfg.speed_limit))

        r_saf = (
            self.cfg.w_col * r_col
            + self.cfg.w_dev * r_dev
            + self.cfg.w_head * r_head
            + self.cfg.w_off * r_off
            + self.cfg.w_obs * r_obs
            + self.cfg.w_spd_pen * r_spd_pen
        )

        r_steer = -(steer ** 2)
        r_jerk = -min(1.0, abs(jerk) / 5.0)
        r_brake = -max(0.0, (ctrl.brake - self.cfg.brake_thr))

        r_com = (
            self.cfg.w_steer * r_steer
            + self.cfg.w_jerk * r_jerk
            + self.cfg.w_brake * r_brake
        )

        r_spd = self._speed_reward(speed)
        r_prog = speed / max(1e-3, self.cfg.speed_limit)
        r_eff = self.cfg.w_spd * r_spd + self.cfg.w_prog * r_prog

        r_total = self.cfg.w_saf * r_saf + self.cfg.w_com * r_com + self.cfg.w_eff * r_eff

        self.prev_speed = speed
        self.prev_acc = acc
        self.prev_time = now

        return {
            "sim_time_s": now,
            "eva_total": r_total,
            "eva_safety": r_saf,
            "eva_comfort": r_com,
            "eva_efficiency": r_eff,
            "speed_mps": speed,
            "dev_m": dev,
            "heading_err_rad": head_err,
            "ttc_s": ttc,
            "obs_dist_m": obs_dist,
            "r_dev": r_dev,
            "r_head": r_head,
            "r_off": r_off,
            "r_obs": r_obs,
            "r_spd_pen": r_spd_pen,
            "r_spd": r_spd,
            "r_prog": r_prog,
        }


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_eval_config(args: argparse.Namespace) -> Config:
    config = Config()

    config.carla_port = int(args.carla_port)
    config.carla_tm_port = int(args.tm_port)
    config.tm_port = config.carla_tm_port

    config.render = bool(args.render)
    config.spectator_mode = str(args.spectator_mode)

    config.random_scenario = False
    config.scenario = "cones"
    config.scenario_pool = ["cones"]
    config.scenario_weights = {"cones": 1.0}
    config.cone_num = int(args.cone_num)
    config.spawn_min_gap_from_cone = float(args.spawn_gap_from_cone)

    config.max_episode_steps = int(max(args.episode_max_steps, args.steps))

    # 与训练脚本默认保持一致（obs=30）
    config.observations_type = "state_lane_obstacles"
    config.obs_obstacle_k = int(args.obs_obstacle_k)
    config.obs_obstacle_range = float(args.obs_obstacle_range)

    config.use_action_dim2 = False
    config.use_yref_in_steer = False
    config.yref_steer_gain = 0.0
    config.use_yref_mapping = False
    config.yref_gain = 0.0
    config.yref_penalty = 0.0

    config.enable_debug_drawing = False
    config.debug_obstacle_gate = False

    return config


def build_agent(env: CarlaGymEnv, args: argparse.Namespace) -> PPOAgent:
    agent = PPOAgent(
        env=env,
        seed=int(args.seed),
        policy_lr=float(args.policy_lr),
        value_lr=float(args.value_lr),
        gamma=float(args.gamma),
        lambda_=float(args.lambda_),
        clip_ratio=float(args.clip_ratio),
        entropy_regularization=float(args.entropy_reg),
        optimization_steps=(int(args.opt_steps_policy), int(args.opt_steps_value)),
        batch_size=int(args.batch_size),
        update_frequency=int(args.update_frequency),
        name=str(args.model_name),
        weights_dir=str(args.weights_dir),
        load=False,
    )
    agent.load_weights()
    agent.network.deterministic = bool(args.deterministic)
    return agent


def predict_action(agent: PPOAgent, state: Any, deterministic: bool) -> np.ndarray:
    """评估脚本内部推理逻辑（不修改公共网络实现）。"""
    state_t = utils.to_tensor(state)

    if not deterministic:
        action, _, _, _, _ = agent.predict(state_t)
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        return np.asarray(action, dtype=np.float32)

    # deterministic 模式：手动走 tanh(base.loc)，避免调用 TransformedDistribution.mean()
    net = agent.network
    eps = tf.constant(1e-6, dtype=tf.float32)

    dist_lat = net.old_policy_lat(state_t, training=False)
    base_lat = net._get_base_normal(dist_lat)
    lat = tf.tanh(base_lat.loc)
    lat = tf.clip_by_value(lat, -1.0 + eps, 1.0 - eps)

    steer = lat[:, 0:1]
    y_ref = lat[:, 1:2]

    dist_lon = net.old_policy_lon(net._as_input_list(state_t) + [y_ref], training=False)
    base_lon = net._get_base_normal(dist_lon)
    lon = tf.tanh(base_lon.loc)
    lon = tf.clip_by_value(lon, -1.0 + eps, 1.0 - eps)

    action = tf.concat([lon, steer, y_ref], axis=1)
    return np.asarray(action.numpy(), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO cones 场景 100-step 评估并记录 CSV")

    parser.add_argument("--steps", type=int, default=100, help="总记录步数")
    parser.add_argument("--episode-max-steps", type=int, default=256, help="单个 episode 最大步数")
    parser.add_argument("--seed", type=int, default=10, help="初始随机种子")
    parser.add_argument("--run-id", type=str, default="", help="run_id（默认自动生成）")
    parser.add_argument("--output", type=str, default="", help="CSV 输出路径")

    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")

    parser.add_argument("--weights-dir", type=str, default="./weights", help="PPO 权重根目录")
    parser.add_argument("--model-name", type=str, default="ppo-carla-obs30", help="模型目录名")

    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--value-lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-reg", type=float, default=0.02)
    parser.add_argument("--opt-steps-policy", type=int, default=4)
    parser.add_argument("--opt-steps-value", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-frequency", type=int, default=2)

    parser.add_argument("--deterministic", type=int, choices=[0, 1], default=1, help="1=确定性推理, 0=随机采样")
    parser.add_argument("--render", action="store_true", help="开启 CARLA 渲染")
    parser.add_argument("--spectator-mode", type=str, default="none")

    parser.add_argument("--obs-obstacle-k", type=int, default=5)
    parser.add_argument("--obs-obstacle-range", type=float, default=50.0)
    parser.add_argument("--cone-num", type=int, default=5, help="cones 场景锥桶数量")
    parser.add_argument("--spawn-gap-from-cone", type=float, default=20.0, help="自车与第一个锥桶的前向距离(米)")
    parser.add_argument("--print-collision", type=int, choices=[0, 1], default=1, help="终端打印碰撞事件")
    parser.add_argument("--debug-eva", type=int, choices=[0, 1], default=1, help="终端打印EVA诊断")
    parser.add_argument("--debug-eva-interval", type=int, default=20, help="每隔多少step打印一次EVA诊断")
    parser.add_argument("--eva-speed-limit", type=float, default=4.0, help="EVA效率项使用的目标限速(建议cones用4.0)")
    parser.add_argument("--eva-obs-dist-crit", type=float, default=8.0, help="EVA近障碍判定阈值(米)")
    parser.add_argument("--eva-ttc-crit", type=float, default=3.0, help="EVA TTC危险阈值(秒)")
    parser.add_argument("--eva-max-dev", type=float, default=2.0, help="EVA车道偏移归一化上限(米)")
    parser.add_argument("--eva-max-head", type=float, default=0.35, help="EVA航向误差归一化上限(弧度)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output.strip() or os.path.join(SCRIPT_DIR, "result", f"cones_eval_{run_id}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("[Eval] run_id:", run_id)
    print("[Eval] output:", output_path)
    print("[Eval] model:", os.path.join(args.weights_dir, args.model_name))
    print("[Eval] cone_num:", int(args.cone_num))
    print("[Eval] spawn_gap_from_cone_m:", float(args.spawn_gap_from_cone))
    print("[Eval] eva_speed_limit:", float(args.eva_speed_limit))
    print("[Eval] eva_obs_dist_crit:", float(args.eva_obs_dist_crit))
    print("[Eval] eva_ttc_crit:", float(args.eva_ttc_crit))

    env = None
    try:
        cfg = build_eval_config(args)
        env = CarlaGymEnv(cfg, logger=None, wandb_run=None)

        agent = build_agent(env, args)
        eva_cfg = EVAConfig()
        eva_cfg.speed_limit = float(args.eva_speed_limit)
        eva_cfg.obs_dist_crit = float(args.eva_obs_dist_crit)
        eva_cfg.ttc_crit = float(args.eva_ttc_crit)
        eva_cfg.max_dev = float(args.eva_max_dev)
        eva_cfg.max_head = float(args.eva_max_head)
        eva = EvaEvaluator(eva_cfg)

        total_logged_steps = 0
        episode_index = 0
        collision_steps = []
        collision_count = 0
        eva_total_vals = []
        eva_safety_vals = []
        eva_comfort_vals = []
        eva_eff_vals = []
        speed_vals = []
        obs_dist_vals = []
        ttc_vals = []
        r_dev_vals = []
        r_head_vals = []
        r_obs_vals = []
        r_spd_vals = []
        r_prog_vals = []

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while total_logged_steps < int(args.steps):
                episode_seed = int(args.seed) + int(episode_index)
                set_all_seeds(episode_seed)

                state = env.reset()
                done = False
                carla_env = env.carla_env
                world = getattr(carla_env, "world", None)
                ego = getattr(carla_env, "ego", None)
                eva.attach(world, ego)

                while (not done) and (total_logged_steps < int(args.steps)):
                    action = predict_action(agent, state, deterministic=bool(args.deterministic))
                    action_env = agent.convert_action(action)

                    next_state, _, done, info = env.step(action_env)
                    info = info or {}

                    carla_env = env.carla_env
                    world = getattr(carla_env, "world", None)
                    ego = getattr(carla_env, "ego", None)
                    if world is not None and ego is not None:
                        eva.world = world
                        eva.ego = ego

                    collision_flag = int(float(info.get("collision", 0.0)) > 0.5)

                    throttle = 0.0
                    steer = 0.0
                    brake = 0.0
                    if ego is not None:
                        try:
                            ctrl = ego.get_control()
                            throttle = float(ctrl.throttle)
                            steer = float(ctrl.steer)
                            brake = float(ctrl.brake)
                        except Exception:
                            pass

                    extra_candidates = None
                    if carla_env is not None and hasattr(carla_env, "_collect_obstacle_candidates"):
                        try:
                            extra_candidates = carla_env._collect_obstacle_candidates()
                        except Exception:
                            extra_candidates = None

                    eva_metrics = eva.tick(collision=bool(collision_flag), extra_candidates=extra_candidates)
                    if not eva_metrics:
                        eva_metrics = {
                            "sim_time_s": 0.0,
                            "eva_total": 0.0,
                            "eva_safety": 0.0,
                            "eva_comfort": 0.0,
                            "eva_efficiency": 0.0,
                            "speed_mps": 0.0,
                            "dev_m": 0.0,
                            "heading_err_rad": 0.0,
                            "ttc_s": 999.0,
                            "obs_dist_m": 999.0,
                        }

                    speed_mps = float(info.get("speed", eva_metrics.get("speed_mps", 0.0)))
                    sim_time_s = float(eva_metrics.get("sim_time_s", 0.0))
                    scenario_name = str(getattr(carla_env, "scenario", "cones"))
                    eva_total_vals.append(float(eva_metrics["eva_total"]))
                    eva_safety_vals.append(float(eva_metrics["eva_safety"]))
                    eva_comfort_vals.append(float(eva_metrics["eva_comfort"]))
                    eva_eff_vals.append(float(eva_metrics["eva_efficiency"]))
                    speed_vals.append(float(speed_mps))
                    obs_dist_vals.append(float(eva_metrics["obs_dist_m"]))
                    ttc_vals.append(float(eva_metrics["ttc_s"]))
                    r_dev_vals.append(float(eva_metrics.get("r_dev", 0.0)))
                    r_head_vals.append(float(eva_metrics.get("r_head", 0.0)))
                    r_obs_vals.append(float(eva_metrics.get("r_obs", 0.0)))
                    r_spd_vals.append(float(eva_metrics.get("r_spd", 0.0)))
                    r_prog_vals.append(float(eva_metrics.get("r_prog", 0.0)))

                    total_logged_steps += 1
                    if collision_flag:
                        collision_count += 1
                        collision_steps.append(total_logged_steps)
                        if bool(args.print_collision):
                            print(
                                f"[Collision] step={total_logged_steps}, episode={episode_index}, "
                                f"sim_time_s={sim_time_s:.2f}, speed_mps={speed_mps:.2f}"
                            )
                    if bool(args.debug_eva) and (total_logged_steps % max(1, int(args.debug_eva_interval)) == 0):
                        print(
                            f"[EVA] step={total_logged_steps} saf={eva_metrics['eva_safety']:.4f} "
                            f"com={eva_metrics['eva_comfort']:.4f} eff={eva_metrics['eva_efficiency']:.4f} "
                            f"speed={speed_mps:.3f} dev={eva_metrics['dev_m']:.3f} "
                            f"obs_dist={eva_metrics['obs_dist_m']:.3f} ttc={eva_metrics['ttc_s']:.3f} "
                            f"(r_dev={eva_metrics.get('r_dev', 0.0):.3f}, r_head={eva_metrics.get('r_head', 0.0):.3f}, "
                            f"r_obs={eva_metrics.get('r_obs', 0.0):.3f}, r_spd={eva_metrics.get('r_spd', 0.0):.3f}, "
                            f"r_prog={eva_metrics.get('r_prog', 0.0):.3f})"
                        )
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "method_id": FIXED_METHOD_ID,
                            "planner_id": FIXED_PLANNER_ID,
                            "scenario": scenario_name,
                            "episode_index": int(episode_index),
                            "seed": int(episode_seed),
                            "step": int(total_logged_steps),
                            "sim_time_s": float(sim_time_s),
                            "collision": int(collision_flag),
                            "speed_mps": float(speed_mps),
                            "throttle": float(throttle),
                            "steer": float(steer),
                            "brake": float(brake),
                            "eva_total": float(eva_metrics["eva_total"]),
                            "eva_safety": float(eva_metrics["eva_safety"]),
                            "eva_comfort": float(eva_metrics["eva_comfort"]),
                            "eva_efficiency": float(eva_metrics["eva_efficiency"]),
                            "dev_m": float(eva_metrics["dev_m"]),
                            "heading_err_rad": float(eva_metrics["heading_err_rad"]),
                            "ttc_s": float(eva_metrics["ttc_s"]),
                            "obs_dist_m": float(eva_metrics["obs_dist_m"]),
                        }
                    )

                    state = next_state

                episode_index += 1
                f.flush()

        print(f"[Eval] 完成，共写入 {total_logged_steps} 条 step 数据。")
        print(f"[Eval] CSV: {output_path}")
        print(f"[Eval] collisions: {collision_count}")
        if collision_steps:
            print(f"[Eval] collision_steps: {collision_steps}")
        else:
            print("[Eval] collision_steps: []")
        if speed_vals:
            arr_speed = np.asarray(speed_vals, dtype=np.float32)
            arr_saf = np.asarray(eva_safety_vals, dtype=np.float32)
            arr_com = np.asarray(eva_comfort_vals, dtype=np.float32)
            arr_eff = np.asarray(eva_eff_vals, dtype=np.float32)
            arr_obs = np.asarray(obs_dist_vals, dtype=np.float32)
            arr_rdev = np.asarray(r_dev_vals, dtype=np.float32) if r_dev_vals else np.zeros((0,), dtype=np.float32)
            arr_rhead = np.asarray(r_head_vals, dtype=np.float32) if r_head_vals else np.zeros((0,), dtype=np.float32)
            arr_robs = np.asarray(r_obs_vals, dtype=np.float32) if r_obs_vals else np.zeros((0,), dtype=np.float32)
            arr_rspd = np.asarray(r_spd_vals, dtype=np.float32) if r_spd_vals else np.zeros((0,), dtype=np.float32)
            arr_rprog = np.asarray(r_prog_vals, dtype=np.float32) if r_prog_vals else np.zeros((0,), dtype=np.float32)

            print(
                "[EvalSummary] "
                f"speed_mean={arr_speed.mean():.4f}, speed_max={arr_speed.max():.4f}, "
                f"moving_ratio(>0.5mps)={(arr_speed > 0.5).mean():.3f}"
            )
            print(
                "[EvalSummary] "
                f"eva_safety_mean={arr_saf.mean():.6f}, eva_comfort_mean={arr_com.mean():.6f}, "
                f"eva_eff_mean={arr_eff.mean():.6f}"
            )
            print(
                "[EvalSummary] "
                f"obs_dist_min={arr_obs.min():.4f}, obs_dist_mean={arr_obs.mean():.4f}"
            )
            if arr_rdev.size > 0:
                print(
                    "[EvalSummary] "
                    f"r_dev_mean={arr_rdev.mean():.6f}, r_head_mean={arr_rhead.mean():.6f}, "
                    f"r_obs_mean={arr_robs.mean():.6f}, r_spd_mean={arr_rspd.mean():.6f}, "
                    f"r_prog_mean={arr_rprog.mean():.6f}"
                )
            if np.all(np.abs(arr_saf) < 1e-5):
                print(
                    "[EvalHint] eva_safety 近乎全0：更像指标未触发（居中/未碰撞/未近障碍），"
                    "不一定是RL坏。重点看 speed_mean 和 obs_dist_min。"
                )
            if arr_speed.mean() < 0.6:
                print("[EvalHint] 更像RL策略问题：车速长期过低（agent基本没学会前进）。")
            elif arr_obs.min() > float(args.eva_obs_dist_crit) and np.abs(arr_saf).mean() < 0.03:
                print("[EvalHint] 更像EVA阈值问题：几乎从未进入危险区，safety自然接近0。")
            elif np.abs(arr_eff).mean() < 0.05 and arr_speed.mean() > 0.8:
                print("[EvalHint] 更像EVA效率尺度问题：请继续下调 --eva-speed-limit（例如 3.0~4.0）。")

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
