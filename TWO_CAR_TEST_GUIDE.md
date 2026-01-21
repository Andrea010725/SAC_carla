# 双车场景测试指南

**日期**: 2026-01-08
**目的**: 测试双车障碍物场景的生成和检测功能
**状态**: ✅ 已配置

---

## 场景配置

### 核心参数 (config.py:59-64)

```python
num_parked_cars = 2                    # 生成2辆车
parked_car_spacing = 8.0               # 间隔8米
parked_car_offset = 0.0                # 横向偏移0米（车道中心）
parked_car_start_distance_min = 12.0  # 第一辆车最小距离12米
parked_car_start_distance_max = 20.0  # 第一辆车最大距离20米
```

### 场景特点

| 特性 | 配置 | 说明 |
|------|------|------|
| **障碍车数量** | 2辆 | 双车场景 |
| **第一辆车距离** | 12-20米随机 | 每次episode不同 |
| **第二辆车距离** | 第一辆+8米 | 固定间隔 |
| **横向位置** | 车道中心 | 与ego同一车道 |
| **生成策略** | 3重保障 | 中心→高位→偏移 |

---

## 场景示意图

### 俯视图

```
    ←←← 对向车道 ←←←  |  →→→ ego车道 →→→
    ─────────────────────────────────────
                         |
                         |  ego起点
                         |    ●
                         |    ↓ 前进
                         |
                         |  12-20米（随机）
                         |    ↓
                         |
                         |  🚗 障碍车1（车道中心）
                         |    ↓
                         |
                         |  8米（固定）
                         |    ↓
                         |
                         |  🚗 障碍车2（车道中心）
                         |
    ─────────────────────────────────────

    车道宽度: 3.5米
    障碍车位置: 车道中心（横向偏移=0）
    ego必须: 变道或减速避让
```

### 距离示例

假设第一辆车在16米:
```
Ego → 16m → 🚗1 → 8m → 🚗2
 0m        16m       24m

总距离: 24米
都在检测范围内（50米）
```

---

## 测试方法

### 方法1: 运行测试脚本（推荐）

```bash
cd /home/ajifang/SAC_carla
python test_two_car_scenario.py
```

**测试内容**:
1. ✅ 环境创建
2. ✅ 障碍车生成（2辆）
3. ✅ 位置验证（12-20米 + 8米）
4. ✅ 检测功能（obstacle_actors）
5. ✅ 观测值验证（非零元素）
6. ✅ 检测函数测试
7. ✅ 运行10步模拟
8. ✅ 总结报告

### 方法2: 运行训练脚本

```bash
python train_ppo_with_wandb.py
```

观察输出中的 `[OBSTACLE DEBUG]` 信息。

---

## 预期输出

### 成功的输出

```
[SPAWN] 使用预定义spawn点:
  - Ego位置: (0.6, -4.5, 0.3)

[OBSTACLE] 双车场景 - 开始生成2辆障碍车:
  - 起始waypoint: (0.6, -4.5, 0.0)
  - 车道类型: Driving
  - 车道宽度: 3.5m
  - 第一辆车目标距离: 16.3m (范围: 12.0-20.0m)
  - 车辆1: 目标距离=16.0m, 位置=(16.5, -4.3)
    ✅ 成功生成！ID=123, 实际位置=(16.5, -4.3, 0.5)
  - 第二辆车间隔: 8.0m
  - 车辆2: 目标距离=24.0m, 位置=(24.4, -4.1)
    ✅ 成功生成！ID=124, 实际位置=(24.4, -4.1, 0.5)

[OBSTACLE DEBUG] 双车场景生成完成:
  - 尝试生成: 2 辆
  - 成功生成: 2 辆
  - 车辆1位置: (16.5, -4.3, 0.5)
  - 车辆2位置: (24.4, -4.1, 0.5)
  - 两车间距: 8.0m
  - 横向偏移: 0.0m (车道中心)
  - 注册到obstacle_actors: 2 个

[OBSTACLE DETECTION] Step 0:
  - 检测范围: 50.0m
  - 候选障碍物数量: 2
  - Ego位置: (0.6, -4.5, 0.2)
  - 障碍物1: 距离=16.2m ✅, 位置=(16.5, -4.3)
  - 障碍物2: 距离=24.2m ✅, 位置=(24.4, -4.1)
  - 观测值非零元素: 6/15
  - 前3个障碍物观测: [0.32 0.0 0.32  0.48 0.0 0.48  0.0 0.0 0.0]

✅ 双车场景配置成功！
```

---

## 障碍物检测函数详解

### 1. _collect_obstacle_candidates()

**位置**: `carla_env.py:1009-1042`

**功能**: 收集候选障碍物

**逻辑**:
```python
def _collect_obstacle_candidates(self):
    # 优先使用注册的obstacle_actors
    if hasattr(self, "obstacle_actors") and self.obstacle_actors:
        return [a for a in self.obstacle_actors if a is not None]

    # Fallback: 扫描世界中的所有车辆
    actors = self.world.get_actors()
    out = []
    for v in actors.filter("vehicle.*"):
        if v.id != self.ego.id:
            out.append(v)
    return out
```

**测试方法**:
```python
candidates = env._collect_obstacle_candidates()
print(f"候选障碍物数量: {len(candidates)}")
# 预期: 2
```

### 2. _compute_obstacle_obs()

**位置**: `carla_env.py:1044-1129`

**功能**: 计算障碍物观测值

**逻辑**:
```python
def _compute_obstacle_obs(self) -> np.ndarray:
    K = 5           # 最多5个障碍物
    R = 50.0        # 检测范围50米

    candidates = self._collect_obstacle_candidates()
    if not candidates:
        return np.zeros((K * 3,), dtype=np.float32)

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

        if dist <= R:  # 50米内
            # ego坐标系下的相对位置
            rel_x = dx * ego_fwd.x + dy * ego_fwd.y  # 前向投影
            rel_y = dx * ego_right.x + dy * ego_right.y  # 横向投影
            items.append((dist, rel_x, rel_y))

    # 按距离排序，取最近的K个
    items.sort(key=lambda x: x[0])
    items = items[:K]

    # 归一化
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

**测试方法**:
```python
obstacle_obs = env._compute_obstacle_obs()
print(f"观测维度: {obstacle_obs.shape}")  # (15,)
print(f"非零元素: {np.count_nonzero(obstacle_obs)}")  # 6
print(f"前6个值: {obstacle_obs[:6]}")
# 预期: [0.32, 0.0, 0.32, 0.48, 0.0, 0.48]
```

### 3. _get_state_obs()

**位置**: `carla_env.py:1131-1200`

**功能**: 组合完整观测

**逻辑**:
```python
def _get_state_obs(self):
    parts = []

    # 1. 基础状态 (9维)
    base_state = self._compute_base_state()
    parts.append(base_state)

    # 2. 车道信息 (6维)
    if self.obs_use_lane:
        lane_obs = self._compute_lane_obs()
        parts.append(lane_obs)

    # 3. 障碍物信息 (15维)
    if self.obs_use_obstacles:
        obstacle_obs = self._compute_obstacle_obs()
        parts.append(obstacle_obs)

    obs = np.concatenate(parts, axis=0).astype(np.float32)
    return obs
```

**观测结构**:
```python
obs = [
    # 基础状态 (9维)
    speed, acc_x, acc_y, yaw_rate, ...

    # 车道信息 (6维)
    lane_dev, sin(heading_err), cos(heading_err), ...

    # 障碍物信息 (15维 = 5个障碍物 × 3)
    obs1_rel_x, obs1_rel_y, obs1_dist,  # 障碍物1
    obs2_rel_x, obs2_rel_y, obs2_dist,  # 障碍物2
    0.0, 0.0, 0.0,                      # 障碍物3（无）
    0.0, 0.0, 0.0,                      # 障碍物4（无）
    0.0, 0.0, 0.0,                      # 障碍物5（无）
]

总维度: 9 + 6 + 15 = 30
```

---

## 观测值解析

### 双车场景示例

假设:
- Ego在 (0, 0)
- 障碍车1在前方16米, 车道中心
- 障碍车2在前方24米, 车道中心

**观测值**:
```python
obstacle_obs = [
    0.32, 0.0, 0.32,  # 障碍物1: 前向16m, 横向0m, 距离16m
    0.48, 0.0, 0.48,  # 障碍物2: 前向24m, 横向0m, 距离24m
    0.0,  0.0, 0.0,   # 障碍物3: 无
    0.0,  0.0, 0.0,   # 障碍物4: 无
    0.0,  0.0, 0.0,   # 障碍物5: 无
]

归一化计算:
- 障碍物1: 16m / 50m = 0.32
- 障碍物2: 24m / 50m = 0.48
```

### 反归一化

```python
# 从观测值恢复实际距离
rel_x_norm = obstacle_obs[0]  # 0.32
rel_y_norm = obstacle_obs[1]  # 0.0
dist_norm = obstacle_obs[2]   # 0.32

# 反归一化
R = 50.0  # 检测范围
rel_x = rel_x_norm * R  # 0.32 * 50 = 16m
rel_y = rel_y_norm * R  # 0.0 * 50 = 0m
dist = dist_norm * R    # 0.32 * 50 = 16m
```

---

## 验证检查清单

### 生成阶段

- [ ] 成功生成2辆车
- [ ] 第一辆车距离在12-20米之间
- [ ] 第二辆车距离 = 第一辆 + 8米
- [ ] 两车间距约8米
- [ ] 横向偏移=0（车道中心）
- [ ] 注册到obstacle_actors: 2个

### 检测阶段

- [ ] `_collect_obstacle_candidates()` 返回2个
- [ ] `_compute_obstacle_obs()` 返回15维数组
- [ ] 非零元素数 = 6 (2辆车 × 3)
- [ ] 观测值合理（0.24-0.48之间）

### 运行阶段

- [ ] Ego能正常移动
- [ ] 观测值随距离变化
- [ ] 碰撞检测正常
- [ ] Episode能正常结束

---

## 常见问题排查

### Q1: 只生成了1辆车

**可能原因**:
- 第二辆车生成位置被占用
- 路径太短，无法前进8米

**排查方法**:
```bash
grep "车辆2" training.log
```

**解决方案**:
- 减小间隔: `parked_car_spacing = 6.0`
- 检查地图是否有足够空间

### Q2: 观测值全为0

**可能原因**:
- obstacle_actors为空
- 检测范围太小
- 障碍车距离超出范围

**排查方法**:
```bash
grep "obstacle_actors数量" training.log
grep "候选障碍物数量" training.log
```

**解决方案**:
- 检查生成是否成功
- 增加检测范围: `obs_obstacle_range = 100.0`

### Q3: 两车间距不是8米

**可能原因**:
- Waypoint路径不是直线
- 步长累积误差

**排查方法**:
```bash
grep "两车间距" training.log
```

**解决方案**:
- 这是正常的，允许±1米误差
- 如果偏差>2米，检查地图路径

---

## 测试脚本使用

### 运行测试

```bash
python test_two_car_scenario.py
```

### 预期输出

```
==========================================
双车场景测试
==========================================

[配置]
  - 场景: parked_obstacles
  - 障碍车数量: 2
  - 第一辆车距离: 12.0-20.0m
  - 车辆间隔: 8.0m

[1] 创建CARLA环境...
✅ 环境创建成功

[2] Reset环境（生成障碍车）...
✅ Reset完成，观测维度: (30,)

[3] Ego车位置: (0.6, -4.5, 0.2)

[4] 检查obstacle_actors列表...
  - obstacle_actors数量: 2
  - 障碍物1: ID=123
      位置=(16.5, -4.3, 0.5)
      距离ego=16.2m
  - 障碍物2: ID=124
      位置=(24.4, -4.1, 0.5)
      距离ego=24.2m
  - 两车间距: 8.0m

[5] 检查观测值...
  - 观测值非零元素: 6/15
✅ 障碍物检测正常！

[6] 测试障碍物检测函数...
  - _collect_obstacle_candidates() 返回: 2 个

[7] 测试_compute_obstacle_obs()函数...
  - 返回维度: (15,)
  - 非零元素: 6/15

[8] 运行10步测试...
  Step 0:
    - Ego位置: (0.6, -4.5)
    - 最近障碍物距离: 16.2m
    - 实际距离障碍物1: 16.2m
    - 实际距离障碍物2: 24.2m

[9] 测试完成！

==========================================
测试总结
==========================================
✅ 障碍车生成数量正确: 2辆
✅ 第一辆车距离正确: 16.2m (12-20m)
✅ 障碍物检测正常: 6/15 非零
✅ 两车间距正确: 8.0m (目标8m)

==========================================
🎉 所有测试通过！场景配置正确！
==========================================
```

---

## 总结

### 配置完成

- ✅ 2辆障碍车
- ✅ 第一辆: 12-20米随机
- ✅ 第二辆: 第一辆+8米
- ✅ 车道中心（横向偏移=0）
- ✅ 三重生成策略
- ✅ 详细调试输出

### 检测函数

- ✅ `_collect_obstacle_candidates()`: 收集候选
- ✅ `_compute_obstacle_obs()`: 计算观测值
- ✅ `_get_state_obs()`: 组合完整观测

### 测试方法

- ✅ 测试脚本: `test_two_car_scenario.py`
- ✅ 训练脚本: `train_ppo_with_wandb.py`
- ✅ 验证清单
- ✅ 问题排查指南

---

**配置日期**: 2026-01-08
**状态**: ✅ 已完成
**下一步**: 运行测试脚本验证
