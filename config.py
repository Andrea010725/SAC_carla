import torch


class Config:
    def __init__(self):
        # ===== Basics =====
        self.seed = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== CARLA =====
        self.carla_version = "0.9.15"
        self.carla_port = 2000
        self.carla_tm_port = 3000

        # ✅ 单 env 很容易高方差；能开就开到 2~4
        self.num_parallel_envs = 2
        self.carla_ports = [2000, 2002]
        self.carla_tm_ports = [3000, 3002]

        self.test_carla_port = 2000
        self.test_carla_tm_port = 8500

        # ===== Simulation Timing =====
        self.fixed_dt = 0.05
        self.frame_skip = 1

        # ===== Rendering / Observation =====
        self.render = True
        self.observations_type = "state"

        # ===== Map / Vehicle / Traffic =====
        self.map_name = "Town05"
        self.vehicle_name = "tesla.cybertruck"

        # ✅ 关键：确保 ego 不是 autopilot（否则你训练出来的东西会被“保底控制”掩盖）
        self.autopilot = False

        # traffic 可以保留给背景车（如果你的 env 是这么用的）
        self.traffic = True
        self.changing_weather_speed = 0.0  # ✅ 先关天气变化，减少分布漂移

        # ===== Traffic Manager Settings =====
        self.tm_synchronous = True
        self.tm_hybrid_physics = True

        # ===== RL dims =====
        # ✅ 你日志里 states 是 [*, 30]，说明你实际 obs 维度是 30
        self.obs_dim = 30

        # ✅ 你现在 PPO action 是 [throttle_brake, steer, y_ref] => 3 维
        self.action_dim = 3

        # ❌ 下面这些 planner selection 字段如果你这次不用，建议直接关掉避免混淆
        self.planner_selection_mode = False
        self.low_level_action_dim = 2
        self.planner_bins = [-2 / 3, 2 / 3]
        self.planner_mode = "RL"

        # ===== Scenario =====
        self.scenario = "parked_obstacles"
        self.spectator_mode = "none"

        # ✅ 为了让 y_ref 真的"有用"，建议打开一点点随机性（否则它学不到分岔决策）
        self.random_scenario = True
        self.scenario_pool = [
            "parked_obstacles",
            "cones",
            # ✅ 新增场景（已实现）
            "pedestrian_crossing",
            "vehicle_opens_door",
            "cut_in",
            "parking_exit",
        ]

        # ===== Parked Obstacles 场景配置 =====
        self.num_parked_cars = 2
        self.parked_car_spacing = 8.0
        # ✅ 很关键：障碍不要永远在车道中心；给一点 lateral offset，否则 y_ref 学不出"选左/选右"
        self.parked_car_offset = 0.4  # (建议 0.3~0.6)
        self.parked_car_start_distance_min = 12.0
        self.parked_car_start_distance_max = 20.0

        # ===== Cones 场景配置 =====
        self.cone_num = 15
        self.cone_step_behind = 3.0
        self.cone_step_lateral = 0.4
        self.cone_z_offset = 0.0
        self.cone_lane_margin = 0.25
        self.cone_min_gap_from_junction = 15.0
        self.cone_grid = 5.0
        self.spawn_min_gap_from_cone = 20.0

        # ===== Pedestrian Crossing 场景配置 =====
        self.pedestrian_distance = 25.0      # 行人距离自车的距离（米）
        self.num_pedestrians = 3             # 行人数量
        self.pedestrian_speed = 1.5          # 行人速度（m/s）
        self.pedestrian_spacing = 2.0        # 行人间距（米）

        # ===== Vehicle Opens Door 场景配置 =====
        self.door_vehicle_distance = 30.0    # 停放车辆距离（米）
        self.door_trigger_distance = 15.0    # 触发距离（米）
        self.door_side = "random"            # 车门侧（"left"/"right"/"random"）

        # ===== Cut In 场景配置 =====
        self.cutin_vehicle_distance = 40.0   # 切入车辆初始距离（米）
        self.cutin_trigger_distance = 30.0   # 触发距离（米）
        self.cutin_direction = "random"      # 切入方向（"left"/"right"/"random"）
        self.cutin_vehicle_speed = 10.0      # 切入车辆速度（m/s）
        self.cutin_lane_change_distance = 15.0  # 变道距离（米）

        # ===== Parking Exit 场景配置 =====
        self.parking_exit_distance = 35.0    # 停车场距离（米）
        self.parking_vehicle_speed = 3.0     # 驶出车辆速度（m/s）
        self.parking_trigger_distance = 20.0 # 触发距离（米）
        self.parking_side = "random"         # 停车场侧（"left"/"right"/"random"）

        # ===== Training =====
        self.max_episode_steps = 500

        # ✅ 这几个 “5e10 / memory_size / alpha / tau” 更像 SAC 的占位，PPO 里建议别用这么夸张的数
        self.train_total_steps = 2_000_000
        self.test_interval_steps = 10_000
        self.save_interval_steps = 50_000

        # ✅ PPO 常见 batch：先用 256（你日志也在用 256/243）
        self.batch_size = 64

        self.gamma = 0.99

        self.record_path = "./result"

        # ===== y_ref mapping =====
        # ✅ 建议做课程：先不映射，让 y_ref 先学成“对纵向有用的条件变量”
        # 训练前 30~50 个 episode：use_yref_mapping = False
        # 之后再打开：True
        self.use_yref_mapping = True  # 先关，稳定后再开
        self.yref_gain = 0.015         # ✅ 先用更小的（0.01~0.02），再慢慢加到 0.03

        # ✅ 给 y_ref 一个很轻的正则，让它别乱飙（否则它容易变成噪声通道）
        self.yref_penalty = 0.002      # (建议 0.001~0.005)

        # ===== Debug =====
        self.log_steer_saturation = True

        # ✅ 你现在 sat_threshold=0.999 太极端，基本永远不触发；建议 0.95~0.98
        self.steer_sat_threshold = 0.97
