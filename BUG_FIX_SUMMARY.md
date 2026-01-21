# PPO 训练错误修复总结

## 问题1: `AttributeError: 'numpy.ndarray' has already numpy() method`

### 错误描述
```
File "/home/ajifang/SAC_carla/planners/rl_agent_only/agents/ppo.py", line 264, in _convert_action
    return a.numpy().astype(np.float32)
AttributeError: 'numpy.ndarray' has already numpy() method
```

### 根本原因
在 `ppo.py:237-264` 的 `_convert_action` 函数中，代码假设输入总是 `tf.Tensor`，但实际上 `action` 可能已经是 `numpy.ndarray`。

### 修复方案
修改 `_convert_action` 函数，让它能同时处理：
- numpy array 输入
- TensorFlow tensor 输入
- 带 batch 维度的输入

**修复代码** (`ppo.py:237-269`):
```python
def _convert_action(a):
    """
    Robustly convert network output action -> env action.
    Supports:
      - tf.Tensor shape (A,)
      - tf.Tensor shape (1, A)
      - np.ndarray shape (A,) or (1, A)
      - list/tuple length A
    Returns:
      - np.ndarray shape (A,), dtype float32
    """
    # ✅ 先检查是否已经是 numpy array
    if isinstance(a, np.ndarray):
        a_np = a.astype(np.float32)
    else:
        # 转成 tensor 处理
        a = tf.convert_to_tensor(a, dtype=tf.float32)
        a_np = a.numpy().astype(np.float32)

    # 处理 batch 维度
    if a_np.ndim == 2:
        a_np = a_np[0]  # (A,)

    # 确保正确 shape
    a_np = a_np.reshape(self.num_actions)

    # clip to env bounds
    eps = 1e-5
    low = self.action_low.numpy() + eps if isinstance(self.action_low, tf.Tensor) else self.action_low + eps
    high = self.action_high.numpy() - eps if isinstance(self.action_high, tf.Tensor) else self.action_high - eps
    a_np = np.clip(a_np, low, high)

    return a_np.astype(np.float32)
```

### 测试结果
✅ 所有测试通过：
- numpy array 输入 ✅
- tensor 输入 ✅
- 带 batch 维度的 numpy array ✅
- 带 batch 维度的 tensor ✅

---

## 问题2: `RuntimeError: ❌ 第一次tick失败，已重试3次: _queue.Empty`

### 错误描述
```
File "/home/ajifang/SAC_carla/carla_base/carla_sync_mode.py", line 109, in _retrieve_data
    item = sensor_queue.get(timeout=min(remaining, 0.5))
_queue.Empty

RuntimeError: ❌ 第一次tick失败，已重试3次:
```

### 根本原因

这是一个 **CARLA 环境重置（reset）时的资源清理和同步问题**：

1. **传感器清理顺序错误**：
   - 原代码先调用 `_cleanup_actors()`（销毁传感器）
   - 再调用 `sync_mode.close()`（尝试停止已销毁的传感器）
   - 导致传感器队列未正确清空

2. **传感器队列堵塞**：
   - 第一个 episode 结束后，传感器队列中有未处理的数据
   - 第二个 episode 的 `reset()` 时队列超时

3. **CARLA 服务器稳定时间不足**：
   - 清理大量 actors 后，CARLA 需要时间处理销毁请求
   - 原来的等待时间（0.3秒）不够

### 修复方案

#### 修复1: 调整清理顺序 (`carla_env.py:362-397`)

**关键改动**：先关闭 `sync_mode`（在传感器还存在时），再清理 actors

```python
def reset(self):
    # ✅ 随机场景选择（如果启用）
    if getattr(self.config, "random_scenario", False):
        scenario_pool = getattr(self.config, "scenario_pool", ["parked_obstacles", "cones"])
        self.scenario = random.choice(scenario_pool)
        print(f"\n[RandomScenario] 本次Episode场景: {self.scenario}")

    # ✅ 关键修复：先关闭 sync_mode（在传感器还存在时），再清理 actors
    if self.sync_mode is not None:
        try:
            # 先清空队列中的残留数据
            for q in self.sync_mode._queues:
                try:
                    while not q.empty():
                        q.get_nowait()
                except:
                    pass

            # 再关闭 sync_mode（会停止传感器监听）
            self.sync_mode.close()
        except Exception as e:
            print(f"[CarlaEnv] ⚠️ 关闭 sync_mode 失败: {e}")
        finally:
            self.sync_mode = None

    # ✅ 给 CARLA 一点时间处理传感器停止
    time.sleep(0.2)

    # 现在可以安全地清理 actors（包括传感器）
    self._cleanup_actors()

    # ✅ 给 CARLA 更多时间来稳定（特别是在清理大量 actors 后）
    time.sleep(0.5)  # 从 0.3 增加到 0.5

    # 强制同步设置，避免某次异常导致 world settings 漂掉
    self._apply_sync_settings(self.fixed_dt)
```

#### 修复2: 增强场景清理 (`carla_env.py:2328-2376`)

**关键改动**：增加 tick 次数和等待时间，让 CARLA 有足够时间处理 actor 销毁

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

### 修复效果

修复后的清理流程：
1. ✅ 先清空传感器队列残留数据
2. ✅ 关闭 sync_mode（停止传感器监听）
3. ✅ 等待 0.2 秒让传感器停止
4. ✅ 清理 actors（包括传感器）
5. ✅ 多次 tick + 等待，让 CARLA 处理销毁
6. ✅ 等待 0.5 秒让 CARLA 稳定
7. ✅ 重新应用同步设置

这样可以确保：
- 传感器在销毁前正确停止监听
- 队列被完全清空
- CARLA 服务器有足够时间处理所有销毁请求
- 下一个 episode 开始时环境处于干净状态

---

## 总结

### 修改的文件
1. `planners/rl_agent_only/agents/ppo.py` - 修复 `_convert_action` 函数
2. `carla_base/carla_env.py` - 修复 `reset()` 和 `_cleanup_actors()` 方法

### 测试状态
- ✅ `convert_action` 函数测试通过
- 🔄 完整训练测试进行中（需要等待第一个 episode 完成并进入第二个 episode）

### 如何验证修复成功
运行训练脚本，观察是否能连续完成多个 episode 而不崩溃：
```bash
python train_ppo_with_wandb.py
```

如果能看到类似以下输出，说明修复成功：
```
Episode 1 terminated after 500 timesteps...
[RandomScenario] 本次Episode场景: parked_obstacles
[CarlaEnv] ✅ 场景 parked_obstacles 已清理
Episode 2 terminated after 500 timesteps...
[RandomScenario] 本次Episode场景: cones
[CarlaEnv] ✅ 场景 cones 已清理
Episode 3 terminated after 500 timesteps...
```

---

## 日期
2026-01-16
