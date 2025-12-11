import torch
import math

    
class Config:
    def __init__(self):     # 
         
        self.seed = 10   # numpy和torch的随机数种子

    #----------环境相应参数------------

        self.max_episode_steps=500
        self.render= False
        self.carla_port= 2000
        self.changing_weather_speed= 0.1
        self.frame_skip= 1
        self.observations_type= 'state'
        self.traffic = True
        self.vehicle_name = 'tesla.cybertruck'
        self.map_name = 'Town05'
        self.autopilot = True

        self.num_parallel_envs = 1
        self.carla_ports= [2000]
        self.carla_tm_ports = [3000]

        self.test_carla_port= 2000
        self.test_carla_tm_port= 8500  # 改用8500端口避免冲突


    #----------------------------------
        self.obs_dim = 9
        self.action_dim = 2
        
        self.train_total_steps = 5e10
        self.test_interval_steps = 2e3
        self.save_interval_steps = 4e4
        self.warmup_steps = 4e2
        self.eval_episodes = 3
        self.memory_size = 8e5
        self.batch_size = 256*4
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4

    #-------------------------------------
        self.record_path = "./result"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === NEW: Planner-selection 模式开关 ===
        self.planner_selection_mode = True  # 使用 SAC 选择 planner，而不是直接输出(-1,1)^2 控制
        # 分界点分析：
        # [-1, -1/3) = 0.33 长度 -> RULE (33%)
        # [-1/3, 1/3) = 0.67 长度 -> IL (67%)  <- 最多被选
        # [1/3, 1] = 0.33 长度 -> RL (33%)
        # 如果想要三个区间等长（各占33%），改为：[-2/3, 0] 或 [-2/3, 2/3]
        self.planner_bins = [-2/3, 2/3]     # 现在 RULE/IL/RL 各占 33% 概率

        # === 迁移到 CARLA 0.9.15 相关（确保路径&端口）===
        self.carla_version = "0.9.15"

        self.tm_synchronous = True
        self.tm_hybrid_physics = True  # 可选：大图场景提升性能

        # === 动作维度变更：1 维 ===
        # 原: self.action_dim = 2
        self.action_dim = 1  # NEW: 只输出一个标量, 用于选择 planner

        # 保存旧值以备切回（可选）
        self.low_level_action_dim = 2  # 低层控制维度，用于 planner 产生控制

        self.render = True  # 打开 pygame 渲染窗口
        self.observations_type = 'state'  # 训练仍用 state，显示相机仅用于可视化
        self.fixed_dt = 0.05  # 20 FPS
        self.planner_mode = 'RULE'  # 显示Rule Planner（可填 'RL' / 'IL' / 'RULE'）

        # config.py 里新增/确认这些字段
        # 使用Town10HD_Opt（当前已加载的地图，避免切换超时）
        self.map_name = "Town10HD_Opt"  # 或 "Town05" （需要等待切换）

        # 选择场景：'plain'=原样空场景、'cones'=放锥桶、'parked_obstacles'=停车障碍（overtaking训练）
        self.scenario = "parked_obstacles"  # 🔧 新增：停车障碍场景（模拟overtaking）

        # 锥桶参数配置
        self.cone_num = 15                    # 锥桶数量
        self.cone_step_behind = 3.0           # 纵向间距（米）
        self.cone_step_lateral = 0.4          # 横向推进步长
        self.cone_z_offset = 0.0              # 高度偏移
        self.cone_min_gap_from_junction = 15.0  # 距路口最小距离
        self.cone_grid = 5.0                  # 采样网格
        self.cone_lane_margin = 0.25          # 车道边界安全距离
        self.spawn_wp_step = 2.0              # spawn回溯步长
        self.spawn_min_gap_from_cone = 20.0   # 与第一个锥桶的最小距离

        # （可选）强制自车初始位姿；若指定，将覆盖场景默认spawn
        # 字典形式：{'x':..., 'y':..., 'z':..., 'yaw':...}
        self.initial_spawn_tf = None  # or {'x': 1.0, 'y': 2.0, 'z': 0.3, 'yaw': 90.0}

        # （可选）观众视角模式：'none'不处理，'fixed'固定后上方，'chase'追尾
        self.spectator_mode = "none"  # or 'fixed' or 'chase'

        # 🔧 新增：停车障碍场景参数（模拟 Overtaking）
        self.num_parked_cars = 4              # 停车障碍数量
        self.parked_car_spacing = 50.0        # 停车间距（米）
        self.parked_car_offset = 1.8          # 停车横向偏移（米，路边）
        self.parked_car_start_distance = 30.0 # 第一辆车距起点距离（米）

        # 如果想用XML场景，需要先创建XML文件，然后取消下面的注释：
        # self.scenario = "cones_xml"
        # self.xml_file = "/path/to/your/waypoints.xml"

