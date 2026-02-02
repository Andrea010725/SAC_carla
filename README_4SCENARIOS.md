# 📦 完整文件清单

## ✅ 已创建的文件

### 1. 核心训练文件
| 文件名 | 大小 | 行数 | 说明 |
|--------|------|------|------|
| `train_ppo_4scenarios.py` | 33KB | 855行 | **主训练脚本**，支持4场景随机切换 |

### 2. 文档文件
| 文件名 | 大小 | 说明 |
|--------|------|------|
| `4SCENARIOS_MODIFICATION_GUIDE.md` | 8.1KB | 详细修改指南，包含场景参数、故障排查 |
| `QUICK_START_4SCENARIOS.md` | 7.5KB | 快速开始指南，包含使用方法、常见问题 |
| `MODIFICATION_SUMMARY.md` | 7.7KB | 修改总结，包含完整的修改清单 |
| `README_4SCENARIOS.md` | - | **本文件**，完整文件清单和使用说明 |

### 3. 工具脚本
| 文件名 | 大小 | 说明 |
|--------|------|------|
| `test_scenarios_comparison.py` | 7.4KB | 场景对比测试工具 |
| `visualize_scenario_results.py` | 14KB | 训练结果可视化工具 |
| `start_4scenarios_training.sh` | 3.2KB | 一键启动脚本（Bash） |

---

## 📂 目录结构

```
SAC_carla/
├── train_ppo_4scenarios.py          # 主训练脚本
├── test_scenarios_comparison.py     # 测试脚本
├── visualize_scenario_results.py    # 可视化脚本
├── start_4scenarios_training.sh     # 启动脚本
│
├── 4SCENARIOS_MODIFICATION_GUIDE.md # 详细指南
├── QUICK_START_4SCENARIOS.md        # 快速开始
├── MODIFICATION_SUMMARY.md          # 修改总结
├── README_4SCENARIOS.md             # 本文件
│
├── carla_base/
│   ├── carla_env.py                 # ⚠️ 需要修改（添加场景触发逻辑）
│   ├── scenario_manager.py          # ✅ 场景实现（无需修改）
│   └── ...
│
├── weights/
│   └── ppo-carla-4scenarios/        # 权重保存目录（自动创建）
│
├── scenario_plots/                  # 可视化图表目录（自动创建）
│   ├── success_rate_comparison.png
│   ├── collision_rate_comparison.png
│   ├── avg_reward_comparison.png
│   ├── success_rate_over_time.png
│   └── performance_radar.png
│
└── training_log_4scenarios.json     # 训练日志（自动生成）
```

---

## 🚀 快速开始（3步）

### 步骤1：启动CARLA服务器
```bash
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen
```

### 步骤2：运行训练（选择一种方式）

**方式A：使用一键脚本（推荐）**
```bash
cd /home/ajifang/SAC_carla
./start_4scenarios_training.sh
```

**方式B：直接运行Python**
```bash
cd /home/ajifang/SAC_carla
python train_ppo_4scenarios.py
```

### 步骤3：查看结果
```bash
# 可视化训练结果
python visualize_scenario_results.py

# 测试模型性能
python test_scenarios_comparison.py
```

---

## 📊 4个场景说明

### 1. ConesScenario（锥桶避让）
- **难度：** ⭐⭐ 简单
- **描述：** 车道内放置15个锥桶，从一侧逐渐向另一侧移动
- **训练目标：** 学会横向避让和平滑转向
- **预期成功率：** 85-95%

### 2. JaywalkerScenario（鬼探头）
- **难度：** ⭐⭐⭐⭐⭐ 非常困难
- **描述：** 行人突然从路边横穿马路
- **训练目标：** 学会紧急制动和障碍物检测
- **预期成功率：** 60-75%
- **特殊要求：** 需要在 `carla_env.py` 添加触发逻辑

### 3. TrimmaScenario（包围突围）
- **难度：** ⭐⭐⭐⭐ 困难
- **描述：** 前方有快车，左右两侧有慢车，需要找gap超车
- **训练目标：** 学会变道决策和安全超车
- **预期成功率：** 70-85%

### 4. ConstructionLaneChangeScenario（施工变道）
- **难度：** ⭐⭐⭐⭐ 困难
- **描述：** 前方施工封道，相邻车道有高密度交通流
- **训练目标：** 学会找gap变道和避让施工区
- **预期成功率：** 70-80%

---

## 🎯 训练配置

### 场景池配置
```python
config.scenario_pool = [
    "cones",                      # 锥桶场景
    "jaywalker",                  # 鬼探头
    "trimma",                     # 包围突围
    "construction_lane_change",   # 施工变道
]
```

### 训练参数
```python
episodes = 300                    # 训练轮数
timesteps = 512                   # 每轮最大步数
batch_size = 256                  # 批次大小
policy_lr = 1e-4                  # 策略学习率
value_lr = 2e-4                   # 价值学习率
```

### 早停条件
```python
TARGET_SUCCESS = 0.80             # 成功率 >= 80%
TARGET_RAN_FULL = 0.80            # 完整运行率 >= 80%
MAX_COLLISION = 0.10              # 碰撞率 <= 10%
NEED_STABLE_WINDOWS = 3           # 连续3个窗口满足
```

---

## 📈 预期训练效果

### 训练时间
- **单个episode：** 30-60秒
- **300 episodes：** 约4-6小时
- **早停触发：** 可能在150-250 episodes

### 性能指标
| 阶段 | Episodes | 成功率 | 碰撞率 | 说明 |
|------|----------|--------|--------|------|
| 初期 | 1-50 | 20-40% | 40-60% | 探索阶段，Jaywalker场景困难 |
| 中期 | 51-150 | 50-70% | 20-30% | 开始学会避让和减速 |
| 后期 | 151-300 | 70-85% | 5-15% | 各场景表现趋于稳定 |

---

## 🔧 工具使用

### 1. 场景对比测试
```bash
python test_scenarios_comparison.py
```

**输出：**
- 每个场景运行10个episodes
- 统计成功率、碰撞率、平均奖励
- 生成对比报告
- 保存结果到 `scenario_comparison_results.json`

**示例输出：**
```
======================================================================
场景对比报告
======================================================================

场景                      成功率       碰撞率       平均奖励        平均步数
----------------------------------------------------------------------
cones                     90.0%       10.0%         245.67         487.3
trimma                    75.0%       20.0%         198.45         456.8
construction_lane_change  72.0%       25.0%         189.23         442.1
jaywalker                 65.0%       30.0%         167.89         398.5
----------------------------------------------------------------------
总体                      75.5%       21.3%
======================================================================
```

### 2. 训练结果可视化
```bash
python visualize_scenario_results.py
```

**生成图表：**
1. `success_rate_comparison.png` - 成功率对比柱状图
2. `collision_rate_comparison.png` - 碰撞率对比柱状图
3. `avg_reward_comparison.png` - 平均奖励对比柱状图
4. `success_rate_over_time.png` - 训练过程成功率变化曲线
5. `performance_radar.png` - 综合性能雷达图

**文本报告：**
- `scenario_analysis_report.txt` - 详细的文本分析报告

### 3. 一键启动脚本
```bash
./start_4scenarios_training.sh
```

**功能：**
- ✅ 自动检查CARLA服务器状态
- ✅ 检查Python环境和必要文件
- ✅ 创建输出目录
- ✅ 启动训练
- ✅ 显示训练提示和下一步操作

---

## ⚠️ 重要提示

### 1. carla_env.py 需要修改

**文件：** `carla_base/carla_env.py`

**位置：** `step()` 方法，约第564行（`self._tick_once()` 之后）

**添加代码：**
```python
# ===== ✅ 新增：场景触发逻辑 =====
if self.scenario_instance:
    # JaywalkerScenario: 检查并触发行人横穿
    if hasattr(self.scenario_instance, 'check_and_trigger'):
        try:
            ego_loc = self.ego.get_location()
            self.scenario_instance.check_and_trigger(ego_loc)
        except Exception as e:
            print(f"⚠️ check_and_trigger failed: {e}")

    # JaywalkerScenario: 更新行人移动
    if hasattr(self.scenario_instance, 'tick_update'):
        try:
            self.scenario_instance.tick_update()
        except Exception as e:
            print(f"⚠️ tick_update failed: {e}")
```

**为什么需要？**
- JaywalkerScenario 的行人需要在每一帧更新位置
- 需要检测自车距离来触发行人开始横穿
- 不修改也能运行，但 Jaywalker 场景的行人不会移动

### 2. 场景难度差异大

不同场景的难度差异很大，建议：
- **先测试 Cones 场景**验证训练是否正常
- **使用课程学习**逐步增加难度
- **调整早停阈值**适应多场景训练

### 3. 训练时间较长

4场景训练比单场景慢约1.5-2倍：
- 可以关闭渲染加快速度：`config.render = False`
- 可以减少场景复杂度（见文档）
- 可以使用更强的硬件

---

## 🐛 故障排查

### 问题1：CARLA服务器崩溃
**解决：**
```bash
# 杀掉旧进程
pkill -9 CarlaUE4

# 重新启动
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen
```

### 问题2：Jaywalker场景行人不移动
**原因：** 缺少场景触发逻辑

**解决：** 在 `carla_env.py` 添加触发逻辑（见上文）

### 问题3：训练不收敛
**可能原因：**
- 场景难度太高
- 学习率不合适
- 早停阈值太严格

**解决：**
- 降低场景难度（减少障碍物数量）
- 调整学习率
- 放宽早停条件

### 问题4：内存不足
**解决：**
```python
# 减少batch_size
agent = PPOAgent(
    ...
    batch_size=128,  # 从256减少到128
    ...
)
```

---

## 📚 文档索引

### 快速查找
- **快速开始：** `QUICK_START_4SCENARIOS.md`
- **详细指南：** `4SCENARIOS_MODIFICATION_GUIDE.md`
- **修改总结：** `MODIFICATION_SUMMARY.md`
- **本文件：** `README_4SCENARIOS.md`

### 按主题查找
| 主题 | 文档 | 章节 |
|------|------|------|
| 如何运行训练 | QUICK_START | "快速开始" |
| 场景参数配置 | MODIFICATION_GUIDE | "场景参数配置" |
| 故障排查 | QUICK_START | "常见问题" |
| 性能优化 | QUICK_START | "性能优化建议" |
| 进阶技巧 | QUICK_START | "进阶技巧" |
| 修改清单 | MODIFICATION_SUMMARY | "已完成的工作" |

---

## 🎓 进阶使用

### 1. 课程学习策略
```python
# 阶段1：简单场景（100 episodes）
config.scenario_pool = ["cones"]

# 阶段2：添加中等难度（100 episodes）
config.scenario_pool = ["cones", "trimma"]

# 阶段3：添加高难度（100 episodes）
config.scenario_pool = ["cones", "trimma", "construction_lane_change"]

# 阶段4：全部场景（100 episodes）
config.scenario_pool = ["cones", "jaywalker", "trimma", "construction_lane_change"]
```

### 2. 场景权重调整
```python
# 增加简单场景的出现频率
config.scenario_pool = [
    "cones",
    "cones",                      # 重复2次
    "jaywalker",
    "trimma",
    "construction_lane_change",
]
```

### 3. 迁移学习
```python
# 从单场景模型开始
agent = PPOAgent(
    env=env,
    name="ppo-carla-obs30",  # 旧模型
    load=True
)
# 然后在4场景上继续训练
```

---

## 📞 获取帮助

### 查看日志
```bash
# 训练日志
cat training_log_4scenarios.json

# CARLA日志
tail -f /home/ajifang/carla/Saved/Logs/CarlaUE4.log
```

### 检查状态
```bash
# 检查CARLA进程
ps aux | grep CarlaUE4

# 检查GPU使用
nvidia-smi

# 检查磁盘空间
df -h
```

### 联系支持
如果遇到问题：
1. 查看相关文档
2. 检查日志文件
3. 查看Wandb面板的错误信息
4. 参考故障排查章节

---

## 📝 更新日志

### v1.0 (2026-01-27)
- ✅ 创建4场景训练脚本
- ✅ 添加场景触发逻辑支持
- ✅ 完善文档和工具
- ✅ 添加可视化和测试脚本

---

## 🎉 总结

您现在拥有完整的4场景训练系统：

**核心文件：**
- ✅ `train_ppo_4scenarios.py` - 主训练脚本
- ✅ `test_scenarios_comparison.py` - 测试工具
- ✅ `visualize_scenario_results.py` - 可视化工具
- ✅ `start_4scenarios_training.sh` - 启动脚本

**文档：**
- ✅ 详细修改指南
- ✅ 快速开始指南
- ✅ 修改总结
- ✅ 完整文件清单（本文件）

**只需：**
1. ⚠️ 在 `carla_env.py` 添加场景触发逻辑（可选但推荐）
2. 🚀 运行 `./start_4scenarios_training.sh` 开始训练

**祝训练顺利！** 🎉

---

**创建时间：** 2026-01-27 23:40
**版本：** v1.0
**作者：** Claude Code
