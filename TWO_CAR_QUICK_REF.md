# 双车场景测试 - 快速参考

## 🎯 场景配置

```python
# config.py
num_parked_cars = 2                    # 2辆车
parked_car_spacing = 8.0               # 间隔8米
parked_car_offset = 0.0                # 车道中心
parked_car_start_distance_min = 12.0  # 第一辆12-20米
parked_car_start_distance_max = 20.0
```

## 📊 场景示意图

```
Ego → 12-20m随机 → 🚗1 → 8m → 🚗2
 0m                16m       24m

都在车道中心，必须避让
```

## ✅ 测试方法

### 方法1: 测试脚本（推荐）
```bash
python test_two_car_scenario.py
```

### 方法2: 训练脚本
```bash
python train_ppo_with_wandb.py
```

## 📈 预期输出

```
[OBSTACLE DEBUG] 双车场景生成完成:
  - 成功生成: 2 辆
  - 车辆1位置: (16.5, -4.3, 0.5)
  - 车辆2位置: (24.4, -4.1, 0.5)
  - 两车间距: 8.0m

[OBSTACLE DETECTION] Step 0:
  - 候选障碍物数量: 2
  - 障碍物1: 距离=16.2m ✅
  - 障碍物2: 距离=24.2m ✅
  - 观测值非零元素: 6/15

✅ 成功！
```

## 🔍 检测函数

### 1. _collect_obstacle_candidates()
```python
candidates = env._collect_obstacle_candidates()
# 返回: 2个障碍物
```

### 2. _compute_obstacle_obs()
```python
obstacle_obs = env._compute_obstacle_obs()
# 返回: (15,) 数组
# 非零: 6个元素 (2辆车 × 3)
```

### 3. 观测值结构
```python
[
  0.32, 0.0, 0.32,  # 障碍物1: 16m前方
  0.48, 0.0, 0.48,  # 障碍物2: 24m前方
  0.0,  0.0, 0.0,   # 障碍物3: 无
  0.0,  0.0, 0.0,   # 障碍物4: 无
  0.0,  0.0, 0.0,   # 障碍物5: 无
]
```

## ✅ 验证清单

### 生成
- [ ] 成功生成2辆车
- [ ] 第一辆: 12-20米
- [ ] 第二辆: 第一辆+8米
- [ ] 两车间距: ~8米
- [ ] 横向偏移: 0米

### 检测
- [ ] obstacle_actors: 2个
- [ ] 候选障碍物: 2个
- [ ] 观测非零: 6/15
- [ ] 观测值合理: 0.24-0.48

## 🐛 常见问题

### Q: 只生成1辆车
**A**: 减小间隔 `parked_car_spacing = 6.0`

### Q: 观测值全为0
**A**: 检查 `obstacle_actors` 是否为空

### Q: 两车间距不是8米
**A**: 允许±1米误差，正常

## 📝 修改文件

- ✅ `config.py` (行60-64)
- ✅ `carla_env.py` (行899-1045)
- ✅ `test_two_car_scenario.py` (新增)
- ✅ `TWO_CAR_TEST_GUIDE.md` (新增)

---

**配置日期**: 2026-01-08
**状态**: ✅ 已完成
**下一步**: 运行测试验证
