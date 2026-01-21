# 🚀 新场景快速参考

## 📋 场景列表

| 场景名称 | 命令 | 难度 |
|---------|------|------|
| 行人过马路 | `pedestrian_crossing` | ⭐⭐ |
| 车门打开 | `vehicle_opens_door` | ⭐⭐⭐ |
| 切入场景 | `cut_in` | ⭐⭐⭐⭐ |
| 停车场出口 | `parking_exit` | ⭐⭐⭐ |

## 🧪 测试命令

```bash
# 测试单个场景
python test_new_scenarios.py --scenario pedestrian_crossing

# 测试所有场景
python test_new_scenarios.py --all

# 自定义观察时长
python test_new_scenarios.py --scenario cut_in --duration 20
```

## 🏋️ 训练命令

```bash
# 使用所有场景训练（默认）
python train_ppo_with_wandb.py

# 修改 config.py 选择特定场景
# self.scenario_pool = ["pedestrian_crossing", "cut_in"]
```

## ⚙️ 关键配置

```python
# config.py

# 场景池（训练时随机选择）
self.scenario_pool = [
    "parked_obstacles",      # 停放车辆
    "cones",                 # 锥桶
    "pedestrian_crossing",   # 行人过马路 ✨ 新
    "vehicle_opens_door",    # 车门打开 ✨ 新
    "cut_in",                # 切入场景 ✨ 新
    "parking_exit",          # 停车场出口 ✨ 新
]

# 行人过马路配置
self.pedestrian_distance = 25.0  # 距离（米）
self.num_pedestrians = 3         # 数量
self.pedestrian_speed = 1.5      # 速度（m/s）

# 车门打开配置
self.door_vehicle_distance = 30.0   # 距离（米）
self.door_trigger_distance = 15.0   # 触发距离（米）

# 切入场景配置
self.cutin_vehicle_distance = 40.0  # 距离（米）
self.cutin_vehicle_speed = 10.0     # 速度（m/s）

# 停车场出口配置
self.parking_exit_distance = 35.0   # 距离（米）
self.parking_vehicle_speed = 3.0    # 速度（m/s）
```

## 📂 修改的文件

- ✅ `carla_base/scenario_manager.py` - 添加了 4 个新场景类
- ✅ `config.py` - 添加了配置参数
- ✅ `test_new_scenarios.py` - 测试脚本（新文件）

## 🎯 训练策略

### 阶段 1：简单场景
```python
self.scenario_pool = ["pedestrian_crossing"]
```

### 阶段 2：混合场景
```python
self.scenario_pool = [
    "parked_obstacles",
    "pedestrian_crossing",
    "vehicle_opens_door",
]
```

### 阶段 3：全场景
```python
self.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",
    "vehicle_opens_door",
    "cut_in",
    "parking_exit",
]
```

## 🔧 调试技巧

```bash
# 查看场景生成日志
python test_new_scenarios.py --scenario pedestrian_crossing

# 检查场景是否注册
python -c "from carla_base.scenario_manager import ScenarioFactory; print(ScenarioFactory.list_scenarios())"

# 验证配置
python -c "from config import Config; c = Config(); print(c.scenario_pool)"
```

## 📞 常见问题

**Q: 场景初始化失败？**
A: 检查 CARLA 服务器是否运行，尝试更换地图

**Q: 行人/车辆生成失败？**
A: 检查 CARLA 版本（需要 0.9.15），检查 blueprint

**Q: 场景太简单/太难？**
A: 调整 config.py 中的参数（距离、速度、数量）

## 🎓 更多场景

想要更多场景？可以继续实现：
- `signalized_junction_left_turn` - 信号灯左转
- `highway_cut_in` - 高速公路切入
- `follow_leading_vehicle` - 跟车场景
- `actor_flow` - 交通流场景

---

**快速开始：**
```bash
# 1. 启动 CARLA
cd /home/ajifang/carla && ./CarlaUE4.sh

# 2. 测试场景
cd /home/ajifang/SAC_carla
python test_new_scenarios.py --all

# 3. 开始训练
python train_ppo_with_wandb.py
```
