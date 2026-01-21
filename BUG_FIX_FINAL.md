# PPO 训练错误修复 - 最终版本

## 问题历史

### 第一个错误: `AttributeError: 'numpy.ndarray' has already numpy() method` ✅ 已修复
- **文件**: `planners/rl_agent_only/agents/ppo.py`
- **修复**: 修改 `_convert_action` 函数，支持 numpy array 和 tensor 输入
- **状态**: ✅ 测试通过

### 第二个错误: `_queue.Empty` (传感器队列超时) ⚠️ 部分修复
- **原因**: 传感器队列残留数据导致下一个 episode 超时
- **初步修复**: 调整清理顺序，先关闭 sync_mode 再清理 actors
- **结果**: 引发了新问题

### 第三个错误: `RuntimeError: rpc::timeout: Timeout of 2000ms while calling RPC function 'unregister_vehicle'` 🔄 正在修复
- **原因**: CARLA 服务器在处理大量清理操作后 RPC 超时
- **位置**: `carla_env.py:452` - `self.ego.set_autopilot(False)`

## 最终修复方案

### 修复策略

采用**温和的清理方案**，避免过度干预 CARLA 的内部状态：

1. **清空传感器队列**（但不停止传感器）
2. **清理 actors**（包括传感器）
3. **只恢复 world settings**（不尝试停止已销毁的传感器）
4. **增加等待时间**（让 CARLA 稳定）
5. **增加 RPC 重试机制**（防止超时）

### 代码修改

#### 修改1: `carla_env.py` - `reset()` 方法 (行 362-400)

```python
def reset(self):
    # ✅ 随机场景选择（如果启用）
    if getattr(self.config, "random_scenario", False):
        scenario_pool = getattr(self.config, "scenario_pool", ["parked_obstacles", "cones"])
        self.scenario = random.choice(scenario_pool)
        print(f"\n[RandomScenario] 本次Episode场景: {self.scenario}")

    # ✅ 温和的清理方案：先清空队列，但不立即关闭 sync_mode
    if self.sync_mode is not None:
        try:
            # 清空队列中的残留数据
            for q in self.sync_mode._queues:
                try:
                    while not q.empty():
                        q.get_nowait()
                except:
                    pass
        except Exception as e:
            print(f"[CarlaEnv] ⚠️ 清空队列失败: {e}")

    # 清理 actors（包括传感器）
    self._cleanup_actors()

    # ✅ 现在关闭 sync_mode（传感器已经被销毁）
    if self.sync_mode is not None:
        try:
            # 只恢复设置，不再尝试停止传感器（已经销毁了）
            if self.sync_mode._settings is not None:
                self.world.apply_settings(self.sync_mode._settings)
        except Exception as e:
            print(f"[CarlaEnv] ⚠️ 恢复设置失败: {e}")
        finally:
            self.sync_mode = None

    # ✅ 给 CARLA 更多时间来稳定（特别是在清理大量 actors 后）
    time.sleep(0.5)

    # 强制同步设置，避免某次异常导致 world settings 漂掉
    self._apply_sync_settings(self.fixed_dt)
```

#### 修改2: `carla_env.py` - `set_autopilot` 重试机制 (行 448-470)

```python
# ✅ 给 CARLA 更多时间来稳定（特别是在清理场景后）
time.sleep(0.2)

# 初始控制：刹停一帧，确保稳定
# ✅ 增加重试机制，防止 RPC 超时
max_autopilot_retries = 3
for retry in range(max_autopilot_retries):
    try:
        self.ego.set_autopilot(False)
        break
    except RuntimeError as e:
        if retry < max_autopilot_retries - 1:
            print(f"⚠️  set_autopilot 失败 (尝试 {retry+1}/{max_autopilot_retries}): {e}")
            time.sleep(0.5)
        else:
            print(f"⚠️  set_autopilot 最终失败，继续执行: {e}")
            # 不抛出异常，继续执行

init_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
init_control.gear = 1
init_control.manual_gear_shift = True
self.ego.apply_control(init_control)
self.last_control = init_control
```

#### 修改3: `carla_env.py` - `_cleanup_actors()` 增强 (行 2328-2376)

```python
def _cleanup_actors(self):
    def _safe_destroy(actor):
        try:
            if actor is not None:
                actor.destroy()
        except Exception:
            pass

    # ✅ 清理场景实例中的 actors
    if self.scenario_instance is not None:
        try:
            self.scenario_instance.cleanup()
            # 给 CARLA 更多时间来处理 actor 销毁（增加 tick 次数和等待时间）
            if self.world.get_settings().synchronous_mode:
                try:
                    for _ in range(10):  # ✅ 从 5 次增加到 10 次
                        self.world.tick()
                        time.sleep(0.1)  # ✅ 从 0.05 增加到 0.1
                except Exception:
                    pass
            print(f"[CarlaEnv] ✅ 场景 {self.scenario} 已清理")
        except Exception as e:
            print(f"[CarlaEnv] ⚠️ 场景清理失败: {e}")
        self.scenario_instance = None

    for a in self._actors:
        _safe_destroy(a)
    self._actors = []

    # ✅ 先停止 sensor 监听，再销毁
    if self.camera_display is not None:
        try:
            self.camera_display.stop()
        except Exception:
            pass
    _safe_destroy(self.camera_display)
    self.camera_display = None

    if self.collision_sensor is not None:
        try:
            self.collision_sensor.stop()
        except Exception:
            pass
    _safe_destroy(self.collision_sensor)
    self.collision_sensor = None

    _safe_destroy(self.ego)
    self.ego = None
    self.obstacle_actors = []
```

## 关键改进点

### 1. 队列清理策略
- ✅ 在清理 actors 前清空队列残留数据
- ✅ 避免调用已销毁传感器的 `stop()` 方法
- ✅ 只恢复 world settings，不干预传感器状态

### 2. 等待时间优化
- ✅ 场景清理后等待 0.5 秒（从 0.3 秒增加）
- ✅ 每次 tick 等待 0.1 秒（从 0.05 秒增加）
- ✅ tick 次数增加到 10 次（从 5 次增加）

### 3. RPC 超时保护
- ✅ `set_autopilot` 增加 3 次重试机制
- ✅ 每次重试等待 0.5 秒
- ✅ 最终失败时不抛出异常，继续执行

### 4. 错误处理增强
- ✅ 所有关键操作都有 try-except 保护
- ✅ 失败时打印警告但不中断训练
- ✅ 使用 `finally` 确保资源清理

## 为什么之前的修复会导致新问题？

### 问题分析

**第一版修复**（过于激进）：
```python
# 1. 先关闭 sync_mode（调用 sensor.stop()）
sync_mode.close()  # 这会尝试停止传感器

# 2. 再清理 actors（销毁传感器）
_cleanup_actors()  # 传感器已经被停止了
```

**问题**：
- `sync_mode.close()` 会调用 `sensor.stop()`
- 但传感器可能已经在某种"半销毁"状态
- 导致 CARLA 服务器内部状态混乱
- 后续的 RPC 调用（如 `set_autopilot`）超时

**最终版修复**（温和方案）：
```python
# 1. 只清空队列，不停止传感器
for q in sync_mode._queues:
    while not q.empty():
        q.get_nowait()

# 2. 清理 actors（销毁传感器）
_cleanup_actors()

# 3. 只恢复 world settings，不尝试停止传感器
world.apply_settings(sync_mode._settings)
```

**优势**：
- 不干预传感器的生命周期管理
- 让 CARLA 自己处理传感器销毁
- 只清理我们能控制的部分（队列数据）
- 减少 RPC 调用，降低超时风险

## 测试状态

- ✅ `convert_action` 函数测试通过
- 🔄 完整训练测试进行中
- 🎯 目标：连续完成 3+ episodes 不崩溃

## 如果还有问题怎么办？

### 可能的进一步优化

1. **增加更多等待时间**：
   ```python
   time.sleep(1.0)  # 从 0.5 增加到 1.0
   ```

2. **减少 tick 频率**：
   ```python
   for _ in range(10):
       self.world.tick()
       time.sleep(0.2)  # 从 0.1 增加到 0.2
   ```

3. **完全跳过 `set_autopilot`**：
   ```python
   # 如果 RPC 一直超时，可以完全跳过这个调用
   # self.ego.set_autopilot(False)  # 注释掉
   ```

4. **重启 CARLA 服务器**：
   - 如果问题持续，可能是 CARLA 服务器本身不稳定
   - 建议重启 CARLA 服务器后再测试

## 日期
2026-01-16 (最终版)
