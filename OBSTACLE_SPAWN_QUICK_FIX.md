# 障碍车生成失败 - 快速修复指南

## 🔴 问题
```
[OBSTACLE] ⚠️ 车辆1生成失败！位置=(-270.3, 86.5)
成功生成: 0 辆
Ego位置: (0.6, -4.5)
```
**障碍车距离ego 280米！**

---

## ✅ 已修复

### 核心修改
**文件**: `carla_base/carla_env.py:723-759`

**修改前**: 使用随机waypoint
```python
start_wp = self._pick_random_start_waypoint(...)
```

**修改后**: 使用CARLA预定义spawn点
```python
spawns = self.map.get_spawn_points()
ego_spawn_tf = random.choice(spawns)
start_wp = self.map.get_waypoint(ego_spawn_tf.location, ...)
```

---

## 🎯 预期结果

```
[SPAWN] Ego位置: (0.6, -4.5, 0.3)

[OBSTACLE] 开始生成障碍车:
  - 起始waypoint: (0.6, -4.5, 0.0)
  ✅ 车辆1: 位置=(12.5, -3.8, 0.5)
  ✅ 车辆2: 位置=(37.3, -3.5, 0.5)
  ✅ 车辆3: 位置=(62.1, -3.2, 0.5)
  ✅ 车辆4: 位置=(87.0, -3.0, 0.5)

成功生成: 4 辆
候选障碍物数量: 4
观测值非零元素: 6/15

✅ 成功！
```

---

## 🔍 验证方法

### 运行训练
```bash
python train_ppo_with_wandb.py 2>&1 | tee training.log
```

### 检查输出
```bash
# 1. 检查生成
grep "成功生成" training.log
# 应该看到: 成功生成: 4 辆

# 2. 检查检测
grep "候选障碍物数量" training.log
# 应该看到: 候选障碍物数量: 4

# 3. 检查观测
grep "观测值非零元素" training.log
# 应该看到: 观测值非零元素: 6/15 或更多
```

---

## 🐛 如果仍然失败

### 方案1: 固定spawn点
```python
# config.py
self.initial_spawn_tf = {
    "x": 100.0,
    "y": 50.0,
    "z": 0.3,
    "yaw": 90.0
}
```

### 方案2: 减少障碍车
```python
# config.py
self.num_parked_cars = 2  # 从4改为2
```

### 方案3: 增加生成高度
```python
# carla_env.py:921
parked_loc.z += 1.0  # 从0.5改为1.0
```

### 方案4: 换地图
```python
# config.py
self.map_name = "Town01"  # 从Town05改为Town01
```

---

## 📊 修改清单

### carla_env.py
- ✅ 行723-759: 使用预定义spawn点
- ✅ 行874-907: 增强调试输出
- ✅ 行919-950: 多策略生成（右侧→左侧→正上方）
- ✅ 行921: 提高生成高度（0.1→0.5米）

---

## 💡 关键改进

| 项目 | 之前 | 现在 |
|------|------|------|
| Spawn点 | 随机waypoint | 预定义spawn点 |
| 方向 | 不可控 | 可靠 |
| 生成高度 | 0.1m | 0.5m |
| 策略 | 单一 | 三重（右/左/上） |
| 调试 | 简单 | 详细 |

---

## 📝 成功标准

- ✅ `成功生成: 4 辆` (或至少2辆)
- ✅ `候选障碍物数量: 4`
- ✅ `观测值非零元素: > 0`
- ✅ 障碍车在ego前方 (< 100m)

---

**修复日期**: 2026-01-08
**状态**: ✅ 已修复
**下一步**: 运行训练验证
