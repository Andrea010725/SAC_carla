# 障碍物检测快速参考

## 🔧 已修复的问题

**问题**: 障碍物检测函数返回0
**原因**: 障碍车没有注册到 `obstacle_actors` 列表
**修复**: 在 `carla_env.py:911` 添加 `self.obstacle_actors.append(vehicle)`

---

## 📊 当前配置

```python
# config.py
num_parked_cars = 4              # 4辆障碍车
parked_car_spacing = 25.0        # 间隔25米
parked_car_offset = 1.8          # 右侧1.8米
parked_car_start_distance = 12.0 # 起点12米
obs_obstacle_range = 50.0        # 检测范围50米
obs_obstacle_k = 5               # 最多检测5个
```

---

## 🚗 障碍车位置

| 车辆 | 距离 | 在检测范围内? |
|------|------|-------------|
| 第1辆 | 12m | ✅ 是 |
| 第2辆 | 37m | ✅ 是 |
| 第3辆 | 62m | ❌ 否 (>50m) |
| 第4辆 | 87m | ❌ 否 (>50m) |

**建议**: 增加检测范围到100米

---

## ✅ 验证方法

### 快速测试
```bash
cd /home/ajifang/SAC_carla
python test_obstacle_detection.py
```

### 预期输出
```
[OBSTACLE DEBUG] 障碍车生成完成:
  - 成功生成: 4 辆
  - 注册到obstacle_actors: 4 个

[OBSTACLE DETECTION] Step 0:
  - 候选障碍物数量: 4
  - 观测值非零元素: 6/15  ← 应该 > 0

✅ 障碍物检测正常！
```

---

## 🐛 故障排查

### 观测值仍为0?

1. **检查生成**
   ```bash
   grep "OBSTACLE DEBUG" training.log
   ```
   应该看到 "成功生成: 4 辆"

2. **检查检测**
   ```bash
   grep "候选障碍物数量" training.log
   ```
   应该看到 "候选障碍物数量: 4"

3. **检查范围**
   如果距离 > 50m，增加检测范围:
   ```python
   self.obs_obstacle_range = 100.0
   ```

---

## 📈 观测值解释

**维度**: 30 = 9 (base) + 6 (lane) + 15 (obstacles)

**障碍物观测** (15维):
```python
[
  rel_x_1, rel_y_1, dist_1,  # 障碍物1
  rel_x_2, rel_y_2, dist_2,  # 障碍物2
  rel_x_3, rel_y_3, dist_3,  # 障碍物3
  rel_x_4, rel_y_4, dist_4,  # 障碍物4
  rel_x_5, rel_y_5, dist_5,  # 障碍物5
]
```

**示例** (障碍物在前方12m, 右侧0.2m):
```python
[0.24, -0.004, 0.24, ...]
 ^^^^   ^^^^^   ^^^^
 前向   横向    距离
```

---

## 🎯 推荐优化

### 1. 增加检测范围
```python
# config.py
self.obs_obstacle_range = 100.0  # 从50.0改为100.0
```

### 2. 调整车辆间隔
```python
# 更密集训练
self.parked_car_spacing = 15.0

# 更稀疏训练
self.parked_car_spacing = 35.0
```

### 3. 关闭调试输出（训练时）
```python
# carla_env.py
# 注释掉或修改频率
if self.episode_steps % 500 == 0:  # 从50改为500
    print(...)
```

---

## 📝 修改文件

- ✅ `carla_base/carla_env.py` (行911, 912-914, 925-932, 1057-1075, 1122-1127)
- ✅ `test_obstacle_detection.py` (新增)
- ✅ `OBSTACLE_GENERATION_DIAGNOSIS.md` (诊断报告)
- ✅ `OBSTACLE_FIX_SUMMARY.md` (修复总结)

---

## 🚀 下一步

1. 运行测试脚本验证修复
2. 如果成功，运行训练脚本
3. 观察agent是否学会避障
4. 根据需要调整检测范围和车辆间隔

---

**修复日期**: 2026-01-08
**状态**: ✅ 已修复，待验证
