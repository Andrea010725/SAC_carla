# 快速验证重启修复 - 每5个Episodes测试

## 🎯 **验证目标**

测试CARLA重启修复是否有效，通过**更频繁的重启**（每5个episodes）快速验证。

## ⏱️ **验证时间**

### 每5个episodes的时间

| 阶段 | 时间 | 说明 |
|------|------|------|
| 训练5个episodes | 125秒 | 25秒×5 |
| CARLA重启 | 111秒 | 约1.9分钟 |
| **总计** | **236秒** | **约4分钟** |

### 验证计划

| 测试目标 | Episodes | 重启次数 | 预计时间 | 说明 |
|---------|----------|---------|---------|------|
| **最小验证** | 10 | 2次 | **8分钟** | 确认重启能成功1次 |
| **基础验证** | 20 | 4次 | **16分钟** | 确认稳定性 |
| **完整验证** | 30 | 6次 | **24分钟** | 充分验证 |

## 🚀 **开始验证**

### 步骤1: 确保CARLA运行

```bash
ps aux | grep CarlaUE4 | grep -v grep

# 如果没运行：
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low -nosound
```

### 步骤2: 开始训练

```bash
cd /home/ajifang/SAC_carla
./start_fast_training.sh
```

### 步骤3: 观察关键时刻

#### ✅ Episode 5（第1次重启）

**期望看到**：
```
Episode 5 terminated after 234 timesteps in 25.3s with reward 89.01.

======================================================================
Episode 5: 重启CARLA服务器（防止资源泄漏）
======================================================================
[1/7] 清理sync_mode...
[2/7] 清理传感器和actors...
[3/7] 关闭客户端连接...
[4/7] 停止CARLA服务器...
[5/7] 清理Python和系统内存...
[6/7] 启动CARLA服务器...
   等待CARLA启动...
   ✅ CARLA端口2000已就绪 (尝试 X/15)
   等待CARLA完全初始化（Town10HD_Opt大地图需要60-90秒）...
   ✅ 等待完成  ← ✅ 关键：不崩溃
[7/7] 重新创建环境...
✅ CARLA重启完成  ← ✅ 关键：成功

2025-12-25 XX:XX:XX - INFO - Starting new episode
Episode 6 terminated after 123 timesteps in 24.5s  ← ✅ 继续训练
```

#### ✅ Episode 10（第2次重启）

**期望看到**：同上，再次成功重启

#### ✅ Episode 15（第3次重启）

**期望看到**：同上，持续稳定

## ✅ **验证成功标准**

### 最小成功（10 episodes）

- ✅ Episode 5重启成功
- ✅ Episode 6-10正常训练
- ✅ Episode 10重启成功

### 完整成功（20+ episodes）

- ✅ 所有重启都成功（Episode 5, 10, 15, 20...）
- ✅ 每次重启后都能继续训练
- ✅ 训练速度稳定（~25秒/episode）
- ✅ 没有出现TimeoutException崩溃

## ❌ **如果失败**

### 症状A: Episode 5重启时崩溃

```
[6/7] 启动CARLA服务器...
   ✅ CARLA端口2000已就绪
   等待CARLA完全初始化（Town10HD_Opt大地图需要60-90秒）...
   ✅ 等待完成
[7/7] 重新创建环境...
terminate called ... TimeoutException  ← ❌ 还是崩溃
```

**解决**：进一步增加等待时间
```python
# train_ppo_with_wandb.py:385
time.sleep(90)  # 从70秒增加到90秒
```

### 症状B: Episode 5训练很慢

```
Episode 5 terminated after 234 timesteps in 25.3s
[重启成功]
Episode 6 terminated after 123 timesteps in 180s  ← ❌ 慢了7倍
```

**原因**：CARLA还没完全就绪就开始训练
**解决**：在创建环境后增加额外等待
```python
# train_ppo_with_wandb.py:390
env.carla_env = CarlaEnv(env.config, 2000, 8000)
time.sleep(5)  # 🔧 从2秒增加到5秒
```

### 症状C: 第1次重启成功，第2次失败

**原因**：CARLA进程没有完全终止
**解决**：增加杀进程后的等待时间
```python
# train_ppo_with_wandb.py:322-330
subprocess.run(['pkill', '-9', 'CarlaUE4'], check=False)
time.sleep(12)  # 从8秒增加到12秒
```

## 📊 **成功后的下一步**

### 如果10个episodes验证成功

**选项A: 继续当前训练**
- 让训练继续运行到100+ episodes
- 观察长期稳定性

**选项B: 调整重启频率**
- 改回每10个episodes重启（减少开销）
- 继续监控稳定性

```python
# train_ppo_with_wandb.py:784
restart_carla_every=10  # 从5改回10
```

**选项C: 切换到小地图**
- 使用Town05减少重启时间
- 每次重启节省50秒

```python
# config.py:66
self.map_name = "Town05"  # 从Town10HD_Opt改为Town05
```

## 🎯 **快速判断**

### ✅ 8分钟后（10 episodes）

如果看到：
```
Episode 5: 重启CARLA服务器
✅ CARLA重启完成
Episode 6 terminated after 123 timesteps in 24.5s
...
Episode 10: 重启CARLA服务器
✅ CARLA重启完成
Episode 11 terminated after 234 timesteps in 25.3s
```

→ **修复成功！** 🎉

如果看到：
```
Episode 5: 重启CARLA服务器
terminate called ... TimeoutException
```

→ **还需要调整** 😕

---

## 🚀 **现在开始验证！**

```bash
./start_fast_training.sh
```

**预计8分钟后就能看到结果！** ⏱️

---

修改时间: 2025-12-25 13:05
关键改动: restart_carla_every = 5（快速验证）
