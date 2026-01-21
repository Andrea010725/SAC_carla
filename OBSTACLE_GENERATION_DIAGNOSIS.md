# 障碍物生成诊断报告

**日期**: 2026-01-08
**场景**: parked_obstacles
**问题**: 障碍物检测函数返回0，需要判断是场景生成问题还是检测函数问题

---

## 1. 当前配置分析

### 1.1 配置参数 (config.py:60-63)

```python
self.num_parked_cars = 4              # 生成4辆车
self.parked_car_spacing = 25.0        # 车辆间隔25米
self.parked_car_offset = 1.8          # 横向偏移1.8米（右侧）
self.parked_car_start_distance = 12.0 # 第一辆车距离起点12米
```

### 1.2 障碍物观测配置 (carla_env.py:63-67)

```python
self.observations_type = "state_lane_obstacles"  # 包含障碍物观测
self.obs_obstacle_k = 5                          # 最多检测5个障碍物
self.obs_obstacle_range = 50.0                   # 检测范围50米
self.obs_use_obstacles = True                    # 启用障碍物观测
```

---

## 2. 障碍物生成逻辑详解

### 2.1 生成流程 (carla_env.py:870-912)

```python
def _place_parked_vehicles(self, start_wp: carla.Waypoint):
    # 步骤1: 从起点前进到第一辆车位置
    steps = int(self.parked_car_start_distance / 2.0)  # 12.0 / 2.0 = 6步
    for _ in range(6):
        cur_wp = cur_wp.next(2.0)  # 每步2米，总共12米

    # 步骤2: 生成4辆车
    for i in range(4):
        # 2a. 计算停车位置（右侧1.8米）
        wp_tf = cur_wp.transform
        right_vec = wp_tf.get_right_vector()
        parked_loc = wp_tf.location + right_vec * 1.8
        parked_loc.z += 0.1  # 抬高0.1米防止穿地

        # 2b. 生成车辆
        vehicle = world.try_spawn_actor(vehicle_bp, parked_tf)
        if vehicle:
            vehicle.set_simulate_physics(False)  # 关闭物理模拟（静态）
            parked_vehicles.append(vehicle)
            self._actors.append(vehicle)  # ✅ 添加到_actors列表

        # 2c. 前进到下一辆车位置（25米）
        distance_covered = 0.0
        while distance_covered < 25.0:
            cur_wp = cur_wp.next(5.0)  # 每步5米
            distance_covered += 5.0

    return parked_vehicles
```

### 2.2 障碍车位置计算

假设ego车起点在waypoint `start_wp` (x=0, y=0):

| 车辆 | 沿路线距离 | 横向偏移 | 绝对位置（近似） |
|------|-----------|---------|----------------|
| 第1辆 | 12m | 右侧1.8m | (12, -1.8) |
| 第2辆 | 37m (12+25) | 右侧1.8m | (37, -1.8) |
| 第3辆 | 62m (12+25+25) | 右侧1.8m | (62, -1.8) |
| 第4辆 | 87m (12+25+25+25) | 右侧1.8m | (87, -1.8) |

**注意**:
- 横向偏移是相对于waypoint的**右侧**（right_vector）
- 在CARLA中，right_vector通常指向车道右侧（副驾驶方向）
- 1.8米的偏移意味着车辆停在车道边缘或路肩

### 2.3 是否在同一车道？

**答案: 不在同一车道！**

```
车道示意图 (俯视图):

    ←←← 对向车道 ←←←  |  →→→ ego车道 →→→
    ─────────────────────────────────────
                         |  ego起点
                         |    ●
                         |    ↓ 前进
                         |
                         |  🚗 (12m, 右侧1.8m)
                         |    ↓
                         |
                         |  🚗 (37m, 右侧1.8m)
                         |    ↓
                         |
                         |  🚗 (62m, 右侧1.8m)
                         |    ↓
                         |
                         |  🚗 (87m, 右侧1.8m)
    ─────────────────────────────────────

    车道宽度通常: 3.5米
    横向偏移: 1.8米 → 车辆在车道右侧边缘或路肩
```

**关键点**:
- 障碍车在ego车道的**右侧边缘**
- 如果车道宽度是3.5米，车辆中心在距离车道中心1.8米处
- 车辆宽度约2米，所以会**部分占用车道**
- ego车如果保持车道中心行驶，会与障碍车有一定距离（约0.8-1.0米）

---

## 3. 障碍物检测逻辑详解

### 3.1 检测流程 (carla_env.py:991-1029)

```python
def _collect_obstacle_candidates(self):
    # 方法1: 使用注册的obstacle_actors（优先）
    if hasattr(self, "obstacle_actors") and self.obstacle_actors:
        return [a for a in self.obstacle_actors if a is not None]

    # 方法2: 扫描世界中的所有actors（fallback）
    actors = self.world.get_actors()
    out = []

    # 2a. 检测所有车辆（排除ego）
    for v in actors.filter("vehicle.*"):
        if v.id != self.ego.id:
            out.append(v)

    # 2b. 检测静态道具（cones, barriers等）
    for p in actors.filter("static.prop.*"):
        tid = p.type_id
        if "trafficcone" in tid or "cone" in tid or "barrier" in tid:
            out.append(p)

    return out
```

### 3.2 障碍物观测计算 (carla_env.py:1031-1080)

```python
def _compute_obstacle_obs(self) -> np.ndarray:
    K = 5           # 最多5个障碍物
    R = 50.0        # 检测范围50米

    candidates = self._collect_obstacle_candidates()
    if not candidates:
        return np.zeros((K * 3,), dtype=np.float32)  # 返回全0

    ego_loc = self.ego.get_location()
    ego_fwd = self.ego.get_transform().get_forward_vector()
    ego_right = self.ego.get_transform().get_right_vector()

    items = []
    for obstacle in candidates:
        obs_loc = obstacle.get_location()

        # 计算相对位置
        dx = obs_loc.x - ego_loc.x
        dy = obs_loc.y - ego_loc.y
        dist = sqrt(dx^2 + dy^2)

        # 只保留范围内的障碍物
        if dist <= R:  # 50米内
            # 计算ego坐标系下的相对位置
            rel_x = dx * ego_fwd.x + dy * ego_fwd.y  # 前向投影
            rel_y = dx * ego_right.x + dy * ego_right.y  # 横向投影
            items.append((dist, rel_x, rel_y))

    # 按距离排序，取最近的K个
    items.sort(key=lambda x: x[0])
    items = items[:K]

    # 归一化并填充
    feats = []
    for dist, rel_x, rel_y in items:
        rel_x_norm = clip(rel_x / R, -1.0, 1.0)
        rel_y_norm = clip(rel_y / R, -1.0, 1.0)
        dist_norm = clip(dist / R, 0.0, 1.0)
        feats.extend([rel_x_norm, rel_y_norm, dist_norm])

    # 不足K个时填充0
    while len(feats) < K * 3:
        feats.extend([0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)
```

---

## 4. 问题诊断

### 4.1 可能的问题

#### 问题1: obstacle_actors列表为空 ⚠️ **最可能**

**现象**:
```python
# carla_env.py:100
self.obstacle_actors: List[carla.Actor] = []  # 初始化为空列表

# carla_env.py:910
self._actors.append(vehicle)  # 添加到_actors，但没有添加到obstacle_actors！
```

**原因**:
- `_place_parked_vehicles()` 只把车辆添加到 `self._actors`
- **没有添加到 `self.obstacle_actors`**
- `_collect_obstacle_candidates()` 优先检查 `obstacle_actors`
- 如果 `obstacle_actors` 为空但存在（hasattr返回True），就不会fallback到扫描world

**验证方法**:
```python
# 在reset()后添加调试输出
print(f"[DEBUG] obstacle_actors: {len(self.obstacle_actors)}")
print(f"[DEBUG] _actors: {len(self._actors)}")
```

#### 问题2: 检测范围不足 ❌ **不太可能**

**配置**: `obs_obstacle_range = 50.0` 米

**障碍车位置**: 12m, 37m, 62m, 87m

**分析**:
- 前3辆车都在50米内 ✅
- 第4辆车在87米，超出范围 ⚠️
- 但至少应该检测到前3辆车

#### 问题3: 观测类型配置错误 ❌ **已排除**

**配置**: `observations_type = "state_lane_obstacles"`

**检查**:
```python
# carla_env.py:80-82
elif self.observations_type in ["state_lane_obstacles", "lane_obstacles", "full"]:
    self.obs_use_lane = True
    self.obs_use_obstacles = True  # ✅ 已启用
```

#### 问题4: 车辆生成失败 ⚠️ **需要验证**

**可能原因**:
- `world.try_spawn_actor()` 返回None（生成失败）
- 位置被占用
- Blueprint不存在

**验证方法**:
```python
# 在_place_parked_vehicles()中添加
vehicle = world.try_spawn_actor(vehicle_bp, parked_tf)
if vehicle:
    print(f"[DEBUG] 成功生成车辆{i+1}: {vehicle.id}")
    parked_vehicles.append(vehicle)
else:
    print(f"[DEBUG] 车辆{i+1}生成失败！")
```

---

## 5. 修复方案

### 方案1: 修复obstacle_actors注册 ✅ **推荐**

**问题根源**: 障碍车没有注册到 `obstacle_actors` 列表

**修改位置**: `carla_env.py:910`

**修改前**:
```python
if vehicle:
    vehicle.set_simulate_physics(False)
    parked_vehicles.append(vehicle)
    self._actors.append(vehicle)  # 只添加到_actors
```

**修改后**:
```python
if vehicle:
    vehicle.set_simulate_physics(False)
    parked_vehicles.append(vehicle)
    self._actors.append(vehicle)
    self.obstacle_actors.append(vehicle)  # ✅ 同时添加到obstacle_actors
```

### 方案2: 修改检测逻辑 ⚠️ **备选**

**问题**: `_collect_obstacle_candidates()` 的优先级逻辑有问题

**修改位置**: `carla_env.py:997`

**修改前**:
```python
if hasattr(self, "obstacle_actors") and self.obstacle_actors:
    # 只有当obstacle_actors非空时才使用
    return [a for a in self.obstacle_actors if a is not None]
```

**修改后**:
```python
if hasattr(self, "obstacle_actors") and len(self.obstacle_actors) > 0:
    # 明确检查长度
    out = []
    for a in self.obstacle_actors:
        if a is not None:
            try:
                _ = a.get_location()  # 验证actor仍然有效
                out.append(a)
            except:
                continue
    if out:  # 只有当有有效actor时才返回
        return out
    # 否则fallback到扫描world
```

### 方案3: 增加调试输出 ✅ **必须**

**目的**: 确认障碍物是否正确生成和检测

**添加位置1**: `carla_env.py:912` (_place_parked_vehicles末尾)

```python
def _place_parked_vehicles(self, start_wp: carla.Waypoint):
    # ... 原有代码 ...

    # ✅ 添加调试输出
    print(f"\n[OBSTACLE DEBUG] 障碍车生成完成:")
    print(f"  - 尝试生成: {self.num_parked_cars} 辆")
    print(f"  - 成功生成: {len(parked_vehicles)} 辆")
    print(f"  - 起始距离: {self.parked_car_start_distance}m")
    print(f"  - 车辆间隔: {self.parked_car_spacing}m")
    print(f"  - 横向偏移: {self.parked_car_offset}m")

    for i, v in enumerate(parked_vehicles):
        loc = v.get_location()
        print(f"  - 车辆{i+1}: ID={v.id}, 位置=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

    return parked_vehicles
```

**添加位置2**: `carla_env.py:1043` (_compute_obstacle_obs开头)

```python
def _compute_obstacle_obs(self) -> np.ndarray:
    K = int(self.obs_obstacle_k)
    R = float(self.obs_obstacle_range)

    candidates = self._collect_obstacle_candidates()

    # ✅ 添加调试输出
    if self.episode_steps % 50 == 0:  # 每50步输出一次
        ego_loc = self.ego.get_location() if self.ego else None
        print(f"\n[OBSTACLE DETECTION DEBUG] Step {self.episode_steps}:")
        print(f"  - 检测范围: {R}m")
        print(f"  - 候选障碍物: {len(candidates)}")
        print(f"  - Ego位置: ({ego_loc.x:.1f}, {ego_loc.y:.1f})" if ego_loc else "  - Ego: None")

        if candidates and ego_loc:
            for i, obs in enumerate(candidates[:5]):
                obs_loc = obs.get_location()
                dist = math.sqrt((obs_loc.x - ego_loc.x)**2 + (obs_loc.y - ego_loc.y)**2)
                print(f"  - 障碍物{i+1}: 距离={dist:.1f}m, 位置=({obs_loc.x:.1f}, {obs_loc.y:.1f})")

    if not candidates or (self.ego is None):
        return np.zeros((K * 3,), dtype=np.float32)

    # ... 原有代码 ...
```

**添加位置3**: `carla_env.py:1080` (_get_state_obs末尾)

```python
def _get_state_obs(self):
    # ... 原有代码 ...

    obs = np.concatenate(parts, axis=0).astype(np.float32)

    # ✅ 添加调试输出
    if self.episode_steps % 50 == 0 and getattr(self, "obs_use_obstacles", False):
        obs_start = self.base_state_dim + self.lane_dim
        obs_end = obs_start + self.obstacle_dim
        obstacle_obs = obs[obs_start:obs_end]

        print(f"\n[OBSERVATION DEBUG] Step {self.episode_steps}:")
        print(f"  - 观测维度: {obs.shape[0]}")
        print(f"  - 障碍物观测范围: [{obs_start}:{obs_end}]")
        print(f"  - 障碍物观测值: {obstacle_obs[:9]}")  # 前3个障碍物
        print(f"  - 非零元素数: {np.count_nonzero(obstacle_obs)}")

    return obs
```

---

## 6. 验证步骤

### 步骤1: 添加调试输出

1. 按照方案3添加所有调试输出
2. 运行训练脚本
3. 观察输出

### 步骤2: 检查生成情况

**预期输出**:
```
[OBSTACLE DEBUG] 障碍车生成完成:
  - 尝试生成: 4 辆
  - 成功生成: 4 辆
  - 起始距离: 12.0m
  - 车辆间隔: 25.0m
  - 横向偏移: 1.8m
  - 车辆1: ID=123, 位置=(12.5, -1.8, 0.1)
  - 车辆2: ID=124, 位置=(37.3, -1.9, 0.1)
  - 车辆3: ID=125, 位置=(62.1, -1.8, 0.1)
  - 车辆4: ID=126, 位置=(87.0, -1.7, 0.1)
```

**如果输出"成功生成: 0 辆"** → 问题4（生成失败）
**如果输出"成功生成: 4 辆"** → 继续步骤3

### 步骤3: 检查检测情况

**预期输出**:
```
[OBSTACLE DETECTION DEBUG] Step 0:
  - 检测范围: 50.0m
  - 候选障碍物: 4
  - Ego位置: (0.0, 0.0)
  - 障碍物1: 距离=12.2m, 位置=(12.5, -1.8)
  - 障碍物2: 距离=37.4m, 位置=(37.3, -1.9)
  - 障碍物3: 距离=62.2m, 位置=(62.1, -1.8)
  - 障碍物4: 距离=87.1m, 位置=(87.0, -1.7)
```

**如果输出"候选障碍物: 0"** → 问题1（obstacle_actors为空）
**如果输出"候选障碍物: 4"** → 继续步骤4

### 步骤4: 检查观测值

**预期输出**:
```
[OBSERVATION DEBUG] Step 0:
  - 观测维度: 30
  - 障碍物观测范围: [15:30]
  - 障碍物观测值: [0.24 -0.04 0.24  0.75 -0.04 0.75  ...]
  - 非零元素数: 9
```

**如果输出"非零元素数: 0"** → 检测逻辑有问题
**如果输出"非零元素数: 9"** → 障碍物检测正常！

---

## 7. 最终诊断结论

### 7.1 最可能的问题

**问题**: `obstacle_actors` 列表为空，导致 `_collect_obstacle_candidates()` 返回空列表

**原因**:
1. `_place_parked_vehicles()` 没有将车辆添加到 `obstacle_actors`
2. `_collect_obstacle_candidates()` 优先检查 `obstacle_actors`
3. 即使 `obstacle_actors` 为空列表，`hasattr()` 仍返回True
4. 不会fallback到扫描world中的所有车辆

**证据**:
- `carla_env.py:100`: `self.obstacle_actors = []` (初始化为空)
- `carla_env.py:910`: 只添加到 `_actors`，没有添加到 `obstacle_actors`
- `carla_env.py:997`: `if hasattr(self, "obstacle_actors") and self.obstacle_actors:`
  - 当 `obstacle_actors = []` 时，`self.obstacle_actors` 为False
  - 但这个判断可能在某些情况下不work

### 7.2 推荐修复方案

**立即修复** (方案1):
```python
# carla_env.py:910
if vehicle:
    vehicle.set_simulate_physics(False)
    parked_vehicles.append(vehicle)
    self._actors.append(vehicle)
    self.obstacle_actors.append(vehicle)  # ✅ 添加这一行
```

**同时添加** (方案3):
- 所有调试输出
- 验证修复效果

**可选优化** (方案2):
- 改进 `_collect_obstacle_candidates()` 的fallback逻辑
- 确保即使 `obstacle_actors` 为空也能检测到障碍物

---

## 8. 预期修复后的效果

### 8.1 观测值变化

**修复前**:
```python
obstacle_obs = [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0]
# 全0，表示没有检测到任何障碍物
```

**修复后** (ego在起点):
```python
obstacle_obs = [
    0.24, -0.04, 0.24,  # 障碍物1: 12m前方，右侧0.2m
    0.75, -0.04, 0.75,  # 障碍物2: 37m前方，右侧0.2m
    1.24, -0.04, 1.24,  # 障碍物3: 62m前方（超出范围，裁剪到1.0）
    0.00, 0.00, 0.00,   # 障碍物4: 未检测到（超出50m范围）
    0.00, 0.00, 0.00,   # 障碍物5: 未检测到
]
```

### 8.2 训练效果改善

**修复前**:
- Agent看不到障碍物
- 无法学习避障行为
- 可能直接撞上障碍车

**修复后**:
- Agent能感知到前方障碍物
- 可以学习提前变道或减速
- 避障成功率提高

---

## 9. 额外建议

### 9.1 增加检测范围

**当前**: `obs_obstacle_range = 50.0` 米

**建议**: `obs_obstacle_range = 100.0` 米

**原因**:
- 第4辆车在87米处，当前范围检测不到
- 增加到100米可以检测到所有4辆车
- 给agent更多反应时间

### 9.2 调整车辆间隔

**当前**: `parked_car_spacing = 25.0` 米

**建议**: 根据训练目标调整
- **密集训练**: 15.0米 (更频繁遇到障碍)
- **稀疏训练**: 35.0米 (更接近真实场景)

### 9.3 添加可视化

在CARLA Spectator中查看障碍车位置：

```python
# 在_place_parked_vehicles()末尾添加
if self.spectator_mode != "none":
    for i, v in enumerate(parked_vehicles):
        loc = v.get_location()
        # 在障碍车上方绘制标记
        self.world.debug.draw_string(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 2.0),
            f"OBS{i+1}",
            draw_shadow=False,
            color=carla.Color(255, 0, 0),
            life_time=1000.0
        )
```

---

## 10. 总结

### 问题根源
障碍车已经正确生成，但**没有注册到 `obstacle_actors` 列表**，导致检测函数找不到它们。

### 修复方法
在 `_place_parked_vehicles()` 中添加一行代码：
```python
self.obstacle_actors.append(vehicle)
```

### 验证方法
添加调试输出，确认：
1. 障碍车生成成功（4辆）
2. 检测到候选障碍物（4个）
3. 观测值非零（9-12个非零元素）

### 预期效果
修复后，agent能够感知到前方12m、37m、62m处的障碍车，并学习避障行为。

---

**下一步**: 按照方案1和方案3修改代码，运行训练并观察调试输出。
