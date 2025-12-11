# Overtaking 能力专项测试指南

## 📋 概述

这个文件夹包含了专门针对**Overtaking（超车能力）**的独立测试环境。

### 什么是Overtaking能力？

Overtaking能力测试自动驾驶系统在遇到前方障碍时的超车决策和执行能力，包括：

- 🚧 **施工障碍** - 识别施工区域并安全绕行
- 🚗 **停车障碍** - 处理路边停车占道情况
- ⚠️ **事故场景** - 识别事故并安全避让
- 🚪 **开车门** - 应对突然开启的车门
- 🛑 **侧方危险** - 处理侧方突发危险

---

## 📂 文件夹结构

```
overtaking/
├── data/                              # 数据文件
│   ├── overtaking.xml                 # Overtaking专用路线定义（45条）
│   └── overtaking_route_ids.txt       # 路线ID列表
├── scripts/                           # 脚本文件
│   ├── extract_overtaking_routes.py   # 路线提取脚本
│   ├── test_overtaking.sh             # 测试执行脚本
│   └── analyze_results.py             # 结果分析脚本
├── results/                           # 测试结果（自动生成）
│   ├── batches/                       # 批次结果
│   │   ├── batch_0.json              # 第1批（路线1-5）
│   │   ├── batch_1.json              # 第2批（路线6-10）
│   │   └── ...
│   ├── overtaking_results.json        # 合并的完整结果
│   └── overtaking_analysis.json       # 分析报告
└── README.md                          # 本文档
```

---

## 🎯 测试内容

### 9种Overtaking场景

| 场景类型 | 中文名 | 路线数 | 难度 |
|---------|-------|-------|------|
| Accident | 事故场景 | 5 | ⭐⭐ |
| AccidentTwoWays | 双向道路事故 | 5 | ⭐⭐⭐ |
| ConstructionObstacle | 施工障碍 | 5 | ⭐⭐ |
| ConstructionObstacleTwoWays | 双向道路施工 | 5 | ⭐⭐⭐ |
| HazardAtSideLane | 侧方危险 | 5 | ⭐⭐ |
| HazardAtSideLaneTwoWays | 双向道路侧方危险 | 5 | ⭐⭐⭐ |
| ParkedObstacle | 停车障碍 | 5 | ⭐ |
| ParkedObstacleTwoWays | 双向道路停车障碍 | 5 | ⭐⭐ |
| VehicleOpensDoorTwoWays | 双向道路开车门 | 5 | ⭐⭐⭐ |

**总计**: 45条测试路线

---

## 🚀 快速开始

### 步骤1: 提取路线（已完成）

路线已经从 `bench2drive220.xml` 中提取好了。如果需要重新提取：

```bash
cd /home/ajifang/b2drive/overtaking
python3 scripts/extract_overtaking_routes.py
```

### 步骤2: 运行测试

```bash
cd /home/ajifang/b2drive/overtaking

# 使用screen后台运行（推荐）
screen -S overtaking_test
./scripts/test_overtaking.sh

# 按 Ctrl+A, D 分离screen
```

### 步骤3: 监控进度

```bash
# 查看已完成的批次数
ls results/batches/*.json | wc -l

# 实时监控
watch -n 10 'echo "已完成批次: $(ls results/batches/*.json 2>/dev/null | wc -l)/9"'

# 查看CARLA日志
tail -f /tmp/carla_overtaking.log

# 重新连接screen
screen -r overtaking_test
```

### 步骤4: 分析结果

测试完成后，运行分析脚本：

```bash
cd /home/ajifang/b2drive/overtaking
python3 scripts/analyze_results.py
```

---

## ⚙️ 配置说明

### 测试参数（test_overtaking.sh）

```bash
BATCH_SIZE=5              # 每批5条路线（稳定性优先）
TOTAL_ROUTES=45           # 总共45条路线
CARLA_PORT=2000           # CARLA服务器端口
CARLA_TM_PORT=8000        # Traffic Manager端口
```

### 修改批次大小

如果想要调整批次大小，编辑 `scripts/test_overtaking.sh` 第33行：

```bash
BATCH_SIZE=5  # 可选: 3（超稳定）, 5（推荐）, 10（快速）
```

**推荐值**:
- `BATCH_SIZE=3`: 最稳定，但测试时间最长（15批）
- `BATCH_SIZE=5`: **推荐**，平衡稳定性和速度（9批）
- `BATCH_SIZE=10`: 快速，但可能不稳定（5批）

---

## ⏱️ 时间估算

### 单条路线

```
平均每条路线: 8-12分钟
├── 场景加载: 1-2分钟
├── 路线执行: 5-8分钟
└── 保存结果: 1分钟
```

### 总时间（45条路线）

| 批次大小 | 批次数 | CARLA重启 | 总时间估算 |
|---------|-------|----------|-----------|
| 3条/批 | 15批 | 15次×2分钟 | ~8小时 |
| **5条/批** | **9批** | **9次×2分钟** | **~7小时** ✅ |
| 10条/批 | 5批 | 5次×2分钟 | ~6小时 |

**推荐**: 使用 5条/批，预计 **6-8小时** 完成

---

## 📊 结果解读

### overtaking_results.json

测试完成后的原始结果：

```json
{
  "total_routes": 45,
  "ability": "Overtaking",
  "records": [
    {
      "route_id": "RouteScenario_1773",
      "status": "Completed",
      "scores": {
        "score_route": 95.5,
        "score_penalty": 0.0
      },
      "infractions": {
        "collisions_pedestrian": [],
        "collisions_vehicle": [],
        "red_light": [],
        ...
      }
    },
    ...
  ],
  "summary": {
    "completed": 40,
    "failed": 5,
    "completion_rate": 88.9
  }
}
```

### overtaking_analysis.json

分析脚本生成的详细报告：

```json
{
  "ability": "Overtaking",
  "overall_score": 85.5,
  "total_routes": 45,
  "success_routes": 38,
  "scenario_stats": {
    "Accident": {
      "total": 5,
      "success": 4,
      "completed": 5,
      "failed": 0
    },
    ...
  },
  "summary": {
    "excellent": 6,    // 优秀场景数（≥90%）
    "good": 2,         // 良好场景数（70-89%）
    "pass": 1,         // 及格场景数（50-69%）
    "need_improve": 0  // 需改进场景数（<50%）
  }
}
```

---

## 🛠️ 故障处理

### 问题1: 测试中断

**现象**: 脚本运行到一半停止

**解决**:
```bash
# 脚本支持断点续传，直接重新运行即可
cd /home/ajifang/b2drive/overtaking
./scripts/test_overtaking.sh

# 会显示: 🔄 检测到中断，从批次 X 恢复
```

### 问题2: CARLA崩溃

**现象**:
```
❌ CARLA进程已终止
⚠️  检测到CARLA崩溃，将在下一批次重启
```

**解决**:
- ✅ 脚本会自动重启CARLA
- ✅ 继续执行下一批次
- ⚠️ 如果频繁崩溃，考虑减小 `BATCH_SIZE` 到 3

### 问题3: 端口占用

**现象**:
```
RuntimeError: trying to create rpc server for traffic manager; bind error
```

**解决**:
```bash
# 清理CARLA相关进程
pkill -9 -f CarlaUE4
pkill -9 -f leaderboard

# 清理端口
for pid in $(lsof -t -i:2000 2>/dev/null); do kill -9 $pid; done
for pid in $(lsof -t -i:8000 2>/dev/null); do kill -9 $pid; done

# 重新运行
./scripts/test_overtaking.sh
```

### 问题4: 某批次总是失败

**现象**: 批次N总是失败

**解决**:
```bash
# 查看该批次包含的路线
batch_num=2  # 假设批次2失败
start_idx=$((batch_num * 5))
end_idx=$((start_idx + 4))

cat data/overtaking_route_ids.txt | tr ',' '\n' | sed -n "$((start_idx+1)),$((end_idx+1))p"

# 手动测试单条路线
python3 /home/ajifang/b2drive/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=data/overtaking.xml \
    --routes-subset=1773 \
    --agent=/home/ajifang/b2drive/leaderboard/team_code/il_agent.py \
    --port=2000
```

---

## 📈 性能优化

### 提高测试速度

1. **增加批次大小** (降低稳定性)
   ```bash
   # 修改 scripts/test_overtaking.sh 第33行
   BATCH_SIZE=10  # 从5改为10
   ```

2. **使用更强的GPU**
   - 确保CARLA使用GPU加速
   - 检查: `watch -n 1 nvidia-smi`

3. **降低CARLA质量**
   ```bash
   # 修改 scripts/test_overtaking.sh 第149行
   nohup ./CarlaUE4.sh -RenderOffScreen -quality-level=Low ...
   ```

### 提高稳定性

1. **减小批次大小**
   ```bash
   BATCH_SIZE=3  # 超稳定模式
   ```

2. **增加CARLA启动等待时间**
   ```bash
   # 修改 scripts/test_overtaking.sh 第169行
   sleep 10  # 改为 sleep 20
   ```

---

## 🔧 自定义场景

### 添加新的Overtaking场景

如果B2D添加了新的Overtaking相关场景：

1. 编辑 `scripts/extract_overtaking_routes.py`，在 `OVERTAKING_SCENARIOS` 列表中添加新场景类型

2. 重新提取路线：
   ```bash
   python3 scripts/extract_overtaking_routes.py
   ```

3. 重新运行测试：
   ```bash
   ./scripts/test_overtaking.sh
   ```

### 单独测试某个场景类型

```bash
# 示例: 只测试ParkedObstacle场景
route_ids=$(grep -B 10 'type="ParkedObstacle"' data/overtaking.xml | \
            grep 'route id=' | \
            grep -oP 'id="\K[0-9]+' | \
            paste -sd,)

python3 /home/ajifang/b2drive/leaderboard/leaderboard/leaderboard_evaluator.py \
    --routes=data/overtaking.xml \
    --routes-subset=$route_ids \
    --agent=/home/ajifang/b2drive/leaderboard/team_code/il_agent.py \
    --checkpoint=results/parked_obstacle_only.json
```

---

## 📚 相关文档

- **完整分类分析**: `/home/ajifang/b2drive/ABILITY_CLASSIFICATION_ANALYSIS.md`
- **批次测试指南**: `/home/ajifang/b2drive/rl_agent/carla-driving-rl-agent-master/il_batched_quick_ref.sh`
- **B2D完整测试**: `/home/ajifang/b2drive/rl_agent/test_il_agent_full_batched.sh`

---

## 🎯 评分标准

### Overtaking能力评分

```
优秀 (⭐⭐⭐):  ≥ 90% - Overtaking能力非常强
良好 (⭐⭐):    70-89% - Overtaking能力较好
及格 (⭐):      50-69% - Overtaking能力基本合格
需改进 (❌):    < 50%  - Overtaking能力需要提升
```

### 各场景评分

每个场景类型（如Accident、ParkedObstacle）单独评分，使用相同标准。

---

## 💡 使用建议

1. ✅ **首次测试**: 使用默认配置（`BATCH_SIZE=5`）
2. ✅ **后台运行**: 始终使用 `screen` 或 `tmux`
3. ✅ **定期检查**: 每小时检查一次进度
4. ✅ **保留结果**: 完成后备份 `results/` 目录
5. ✅ **对比分析**: 多次测试后对比 `overtaking_analysis.json`

---

## 🎉 完成标志

测试完成后，你应该看到：

```
✅ Overtaking能力测试完成!

📊 结果文件:
   - 合并结果: /home/ajifang/b2drive/overtaking/results/overtaking_results.json
   - 批次结果: /home/ajifang/b2drive/overtaking/results/batches/batch_*.json

🎉 Overtaking能力测试完成!
```

运行分析后，你会得到：

```
🏆 总体表现
===============================================================

总路线数: 45
成功路线: 38
Overtaking能力得分: 84.44%

👍 评级: ⭐⭐ 良好 - Overtaking能力较好
```

---

## 📞 问题反馈

如果遇到问题或有改进建议，请检查：

1. CARLA日志: `/tmp/carla_overtaking.log`
2. 批次结果: `results/batches/batch_X.json`
3. 路线定义: `data/overtaking.xml`

---

**创建日期**: 2024-12-10
**测试路线数**: 45条
**场景类型数**: 9种
**预计测试时间**: 6-8小时
