# 🐛 问题修复：训练场景池被覆盖

## ❌ 问题描述

**你发现的问题是对的！** 训练时只使用了 2 个场景（`parked_obstacles` 和 `cones`），而不是预期的 6 个场景。

---

## 🔍 问题根源

在 `train_ppo_with_wandb.py` 的第 **701 行**，场景池被硬编码覆盖了：

### **修复前（错误）：**
```python
# train_ppo_with_wandb.py 第 701 行
config.scenario_pool = ["parked_obstacles", "cones"]  # ← 只有 2 个场景！
```

这行代码覆盖了 `config.py` 中定义的 6 个场景配置。

---

## ✅ 修复方案

### **修复后（正确）：**
```python
# train_ppo_with_wandb.py 第 701-708 行
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",   # ✅ 新增：行人过马路
    "vehicle_opens_door",    # ✅ 新增：车门打开
    "cut_in",                # ✅ 新增：切入场景
    "parking_exit",          # ✅ 新增：停车场出口
]  # 场景池（6个场景）
```

---

## 📊 修复前后对比

### **修复前：**
| 场景 | 会运行吗？ |
|------|-----------|
| parked_obstacles | ✅ 会 |
| cones | ✅ 会 |
| pedestrian_crossing | ❌ **不会** |
| vehicle_opens_door | ❌ **不会** |
| cut_in | ❌ **不会** |
| parking_exit | ❌ **不会** |

**总计：只有 2 个场景**

### **修复后：**
| 场景 | 会运行吗？ |
|------|-----------|
| parked_obstacles | ✅ 会 |
| cones | ✅ 会 |
| pedestrian_crossing | ✅ **会** |
| vehicle_opens_door | ✅ **会** |
| cut_in | ✅ **会** |
| parking_exit | ✅ **会** |

**总计：6 个场景** ✅

---

## 🎯 验证修复

### **方法 1：检查代码**
```bash
grep -A 10 "config.scenario_pool" train_ppo_with_wandb.py
```

应该看到：
```python
config.scenario_pool = [
    "parked_obstacles",
    "cones",
    "pedestrian_crossing",   # ✅ 新增
    "vehicle_opens_door",    # ✅ 新增
    "cut_in",                # ✅ 新增
    "parking_exit",          # ✅ 新增
]
```

### **方法 2：运行训练并观察日志**
```bash
python train_ppo_with_wandb.py
```

你应该会看到不同的场景名称：
```
Episode 1:
[RandomScenario] 本次Episode场景: cones

Episode 2:
[RandomScenario] 本次Episode场景: pedestrian_crossing  ← ✅ 新场景

Episode 3:
[RandomScenario] 本次Episode场景: cut_in  ← ✅ 新场景

Episode 4:
[RandomScenario] 本次Episode场景: parking_exit  ← ✅ 新场景
```

---

## 📝 为什么会有这个问题？

### **原因分析：**

1. **config.py 中定义了 6 个场景**
   ```python
   # config.py
   self.scenario_pool = [
       "parked_obstacles",
       "cones",
       "pedestrian_crossing",
       "vehicle_opens_door",
       "cut_in",
       "parking_exit",
   ]
   ```

2. **但 train_ppo_with_wandb.py 中覆盖了配置**
   ```python
   # train_ppo_with_wandb.py
   config = Config()  # ← 先读取 config.py 的配置
   config.scenario_pool = ["parked_obstacles", "cones"]  # ← 然后覆盖！
   ```

3. **结果：只使用了 2 个场景**

### **为什么之前没发现？**

可能是因为：
- 这行代码是在添加新场景之前写的
- 当时只实现了 2 个场景
- 添加新场景后忘记更新这里

---

## ✅ 现在的状态

### **修复完成！**

- ✅ `train_ppo_with_wandb.py` 已更新
- ✅ 场景池包含 6 个场景
- ✅ 所有新场景都会被使用
- ✅ 可以开始训练

### **训练时会使用的场景：**

| # | 场景名称 | 难度 | 类型 |
|---|---------|------|------|
| 1 | parked_obstacles | ⭐⭐ | 静态车辆 |
| 2 | cones | ⭐⭐ | 静态锥桶 |
| 3 | pedestrian_crossing | ⭐⭐ | 动态行人 ✨ |
| 4 | vehicle_opens_door | ⭐⭐⭐ | 半动态车辆 ✨ |
| 5 | cut_in | ⭐⭐⭐⭐ | 动态车辆 ✨ |
| 6 | parking_exit | ⭐⭐⭐ | 动态车辆 ✨ |

---

## 🚀 下一步

### **现在可以开始训练了！**

```bash
python train_ppo_with_wandb.py
```

训练时你会看到 6 个不同的场景在不同的 episode 中出现。

---

## 📞 如何避免类似问题？

### **建议：**

1. **不要在训练脚本中硬编码场景池**
   - ❌ 不好：`config.scenario_pool = ["parked_obstacles", "cones"]`
   - ✅ 好：使用 `config.py` 中的默认值

2. **如果需要覆盖，添加注释说明**
   ```python
   # 临时测试：只使用简单场景
   # config.scenario_pool = ["parked_obstacles", "cones"]

   # 正式训练：使用所有场景
   config.scenario_pool = [
       "parked_obstacles",
       "cones",
       "pedestrian_crossing",
       "vehicle_opens_door",
       "cut_in",
       "parking_exit",
   ]
   ```

3. **使用命令行参数控制场景**
   ```python
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--scenarios', nargs='+', default=None)
   args = parser.parse_args()

   if args.scenarios:
       config.scenario_pool = args.scenarios
   # 否则使用 config.py 中的默认值
   ```

---

## ✅ 总结

- ❌ **问题：** 场景池被硬编码为只有 2 个场景
- ✅ **修复：** 更新为包含所有 6 个场景
- ✅ **验证：** 所有场景都会被使用
- ✅ **状态：** 可以开始训练

**感谢你发现这个问题！现在训练会使用所有 6 个场景了！** 🎉

---

**修复时间：** 2026-01-12
**修复文件：** `train_ppo_with_wandb.py` (第 701-708 行)
**影响：** 训练场景从 2 个增加到 6 个
