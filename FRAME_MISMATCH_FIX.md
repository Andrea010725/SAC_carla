# Frame Mismatch Timeout 错误修复

## 问题描述

```
_queue.Empty: _retrieve_data timed out after 2.0s (frame mismatch detected)
```

错误发生在 `carla_sync_mode.py:75`，在 `env.reset()` 的第一次 `tick()` 调用时。

## 根本原因

1. **传感器初始化延迟**：创建 `CarlaSyncMode` 后，传感器（camera、collision）需要一点时间注册监听队列
2. **第一帧timeout太短**：默认 `timeout=2.0` 秒不足以等待传感器数据到达
3. **帧号不匹配**：`world.tick()` 返回的帧号与传感器数据帧号不一致时，会一直等待直到超时

## 修复方案

### 1. 增加 `_tick_once()` 的 timeout 参数

**文件**: `carla_base/carla_env.py:423`

```python
# 修改前
def _tick_once(self):
    ...
    ret = self.sync_mode.tick(timeout=2.0)

# 修改后
def _tick_once(self, timeout=2.0):
    ...
    ret = self.sync_mode.tick(timeout=timeout)
```

### 2. 第一次 tick 使用更长的 timeout

**文件**: `carla_base/carla_env.py:301`

```python
# 初始一帧（让档位设置生效）
# 🔧 第一次tick使用更长的timeout（10秒），因为传感器需要初始化
self._tick_once(timeout=10.0)
```

**原因**：
- 第一次 tick 时，传感器刚刚创建，数据管道还在建立
- 10秒的timeout给足够时间让传感器数据到达
- 后续的 tick 仍然使用默认的 2秒（step中调用）

### 3. 给传感器注册时间

**文件**: `carla_base/carla_env.py:270-272`

```python
# CarlaSyncMode
fps = int(round(1.0 / self.fixed_dt))
if self.render_display and self.camera_display is not None:
    self.sync_mode = CarlaSyncMode(self.world, self.camera_display, fps=fps)
else:
    self.sync_mode = CarlaSyncMode(self.world, fps=fps)

# 🔧 给传感器一点时间注册监听（避免第一帧丢失）
import time
time.sleep(0.1)
```

**原因**：
- `CarlaSyncMode.__init__()` 中会调用 `sensor.listen()` 注册回调
- 100ms 的延迟确保所有传感器的监听队列都准备好

## 为什么之前没遇到这个问题？

可能的原因：
1. **之前 render=False**：没有 camera 传感器，只有 collision 传感器（数据更快）
2. **硬件/网络变化**：系统负载、CARLA版本更新等导致传感器延迟增加
3. **代码修改累积**：多次修改后的副作用

## 验证修复

```bash
# 1. 清理环境
./clean_memory.sh

# 2. 启动CARLA
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low -nosound

# 3. 运行训练
cd /home/ajifang/SAC_carla
./start_ppo_training_wandb.sh
```

## 预期结果

- ✅ Pygame窗口启动
- ✅ 第一次 `env.reset()` 成功（不超时）
- ✅ 训练正常进行，没有 frame mismatch 错误
- ✅ 后续的 reset 也能正常工作

## 修复时间

2025-12-25 00:20

## 相关文件

- `carla_base/carla_env.py` - 主要修改
- `carla_base/carla_sync_mode.py` - 已有帧匹配逻辑，无需修改
- `train_ppo_with_wandb.py` - 之前修复了 render=True

---

**总结**: 问题出在第一次tick的timeout太短 + 传感器初始化需要时间。通过增加第一次tick的timeout到10秒，并在sync_mode创建后等待0.1秒，问题应该得到解决。
