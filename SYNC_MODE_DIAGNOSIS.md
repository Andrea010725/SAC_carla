# Sync Mode 超时问题诊断指南

## 当前修复状态

### 修复1: 智能的 _retrieve_data 实现

```python
def _retrieve_data(self, sensor_queue, timeout):
    start_time = time.time()
    attempts = 0

    while True:
        elapsed = time.time() - start_time
        remaining = timeout - elapsed

        if remaining <= 0:
            # 超时，打印诊断信息
            print(f"❌ Timeout after {timeout}s, {attempts} attempts")
            raise queue.Empty(...)

        wait_time = min(remaining, 1.0)  # 每次最多等1秒

        try:
            data = sensor_queue.get(timeout=wait_time)
            attempts += 1
            if data.frame == self.frame:
                return data  # ✅ 找到匹配的帧
            else:
                print(f"⚠ Frame mismatch: got {data.frame}, expected {self.frame}")
                # 丢弃旧帧，继续
        except queue.Empty:
            # 队列为空，继续等待
            continue
```

**特点**：
- ✅ 总时间控制（不会超过timeout）
- ✅ 能处理队列为空的情况
- ✅ 能清理旧帧（帧号不匹配）
- ✅ 有详细的调试输出

## 诊断信息解读

### 场景1: 正常运行
```
（无输出或很少输出）
```
- 传感器数据快速到达（< 0.5秒）
- 帧号匹配
- 训练正常

### 场景2: 偶尔慢但成功
```
[CarlaSyncMode] ✓ Got frame 123 after 1.20s, 3 attempts
```
- 传感器延迟较高（> 0.5秒）
- 可能有旧帧需要清理（3次尝试）
- 最终成功

### 场景3: 帧号不匹配
```
[CarlaSyncMode] ⚠ Frame mismatch: got 122, expected 123, discarding...
[CarlaSyncMode] ⚠ Frame mismatch: got 122, expected 123, discarding...
[CarlaSyncMode] ✓ Got frame 123 after 0.80s, 5 attempts
```
- 队列中有旧帧（122）
- 需要清理多次才拿到正确的帧（123）
- 最终成功

### 场景4: 传感器数据丢失
```
[CarlaSyncMode] ⏳ Still waiting for data... (1.0s elapsed, expected frame=123)
[CarlaSyncMode] ⏳ Still waiting for data... (2.0s elapsed, expected frame=123)
...
[CarlaSyncMode] ❌ Timeout after 10.0s, 0 attempts, expected frame=123
```
- **队列始终为空**（0次尝试）
- 传感器没有发送任何数据
- **可能原因**：
  - 传感器被销毁了
  - 传感器监听未注册
  - CARLA服务器卡住了

### 场景5: 世界tick失败
```
（在 world.tick() 处卡住，没有后续输出）
```
- CARLA服务器没有响应
- **可能原因**：
  - CARLA进程崩溃
  - CARLA进程僵死
  - 网络连接问题

## 可能的根本原因

### 原因A: 传感器被过早销毁
- `_cleanup_actors()` 在reset中可能销毁了旧传感器
- 但sync_mode还持有旧传感器的队列引用
- **解决**: 确保 sync_mode 在传感器销毁前清理

### 原因B: Collision传感器问题
- Collision传感器只在碰撞时发送数据？（不确定）
- **解决**: 检查是否应该只同步camera，不同步collision

### 原因C: CARLA服务器负载过高
- 渲染800x600图像 + 物理模拟 + 场景复杂度
- CPU/GPU资源不足导致延迟
- **解决**: 降低分辨率、简化场景、降低画质

### 原因D: 同步模式配置错误
- `fixed_delta_seconds` 与实际fps不匹配
- Traffic Manager未正确同步
- **解决**: 检查world settings

## 下一步行动

### 立即执行
```bash
# 1. 清理环境
./clean_memory.sh

# 2. 启动CARLA
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low -nosound

# 3. 运行训练（观察调试输出）
cd /home/ajifang/SAC_carla
./start_ppo_training_wandb.sh 2>&1 | tee debug.log
```

### 观察重点

1. **第一次reset**：
   - 是否有 "⏳ Still waiting for data..." 输出？
   - 等了多久？
   - 最终成功还是失败？

2. **第一次step**：
   - 是否立即超时？
   - 如果超时，是 0 attempts 还是有多次尝试？

3. **CARLA日志**：
   - 检查CARLA终端是否有错误
   - 是否有"sensor destroyed"或"connection lost"

### 根据输出采取行动

#### 如果看到 "0 attempts"
→ 传感器数据根本没进队列
→ 检查传感器监听是否正确注册

#### 如果看到 "多次 frame mismatch"
→ 队列中有大量旧帧
→ 可能是之前的 episode 没清理干净

#### 如果看到 "⏳ Still waiting"
→ 传感器处理太慢
→ 考虑降低分辨率或增加timeout

## 修复时间

2025-12-25 00:50

---

**当前状态**: 已添加详细调试输出，等待测试结果来确定真正的根本原因。
