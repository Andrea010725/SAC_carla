import glob
import os
import sys
import queue
# sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
# )
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")

import carla


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        """从队列中提取数据，设置更小的单次等待超时，防止死锁"""
        frame_timeout = min(timeout / 10.0, 0.1)  # 每次等待最多 100ms
        max_attempts = int(timeout / frame_timeout) + 5

        for attempt in range(max_attempts):
            try:
                data = sensor_queue.get(timeout=frame_timeout)
                if data.frame == self.frame:
                    return data
                # 如果帧不匹配，数据会丢弃，继续等待下一帧
            except queue.Empty:
                # 继续等待，这是正常的
                continue
            except Exception as e:
                print(f"[CarlaSyncMode._retrieve_data] Error: {e}")
                raise

        # 超过最大尝试次数，抛出超时异常
        raise queue.Empty(f"_retrieve_data timed out after {timeout}s (frame mismatch detected)")

    def destroy(self):
        """显式清理资源：恢复世界设置，清理队列和监听器"""
        try:
            # 恢复原始世界设置
            if self._settings is not None:
                self.world.apply_settings(self._settings)
        except Exception as e:
            print(f"[CarlaSyncMode] Warning: Failed to restore settings: {e}")

        # 清理所有队列中的数据（避免内存泄漏）
        for q in self._queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[CarlaSyncMode] Warning: Failed to clear queue: {e}")

        # 移除所有传感器的监听回调
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception as e:
                print(f"[CarlaSyncMode] Warning: Failed to stop sensor: {e}")

        self._queues.clear()
        self.sensors = ()

    def __del__(self):
        """析构函数：确保资源被释放"""
        try:
            self.destroy()
        except Exception:
            pass
