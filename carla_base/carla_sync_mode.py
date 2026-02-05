import sys
import queue
import time

sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla


class CarlaSyncMode:
    """
    Context manager to synchronize output from different sensors.

    Usage:
        with CarlaSyncMode(world, cam, lidar, fps=20) as sync:
            data = sync.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors

        fps = kwargs.get("fps", 20)
        self.delta_seconds = 1.0 / float(fps)
        # ✅ 训练时可关闭渲染，显著降低 CARLA 内存与显存压力
        # 由外部传入 no_rendering_mode=True 时启用
        self.no_rendering_mode = bool(kwargs.get("no_rendering_mode", False))

        self.frame = None
        self._settings = None
        self._queues = []
        self._stash = []   # one-slot stash per queue to hold "future frame" data
        self._first_tick = True  # ✅ 标记第一次 tick

        self.start()

    def __enter__(self):
        return self

    def start(self):
        # Save current settings
        self._settings = self.world.get_settings()

        # Apply sync settings
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = self.delta_seconds
        # ✅ 根据训练/评估模式决定是否渲染
        # 训练时建议 no_rendering_mode=True，减少断连
        new_settings.no_rendering_mode = self.no_rendering_mode

        # apply_settings returns frame id
        self.frame = self.world.apply_settings(new_settings)

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)
            self._stash.append(None)

        # world snapshot queue
        make_queue(self.world.on_tick)

        # sensor queues
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout=1.0):
        """Tick the world and get synchronized data from world + sensors."""
        self.frame = self.world.tick()

        data = []
        for i, q in enumerate(self._queues):
            try:
                item = self._retrieve_data(i, q, timeout=timeout)
                data.append(item)
            except queue.Empty:
                # ✅ 第一次 tick 时，传感器可能还没准备好，返回 None
                if self._first_tick and i > 0:  # i=0 是 world.on_tick，必须有数据
                    data.append(None)
                else:
                    raise

        # ✅ 第一次 tick 完成后，重置标志
        if self._first_tick:
            self._first_tick = False

        # Basic sanity check: all frames should match self.frame (跳过 None)
        bad = [x.frame for x in data if x is not None and getattr(x, "frame", None) != self.frame]
        if bad:
            raise RuntimeError(
                f"[CarlaSyncMode] Frame mismatch after tick: expected={self.frame}, got={bad}"
            )
        return data

    def _retrieve_data(self, idx, sensor_queue, timeout):
        """
        Robust frame alignment:
        - If stash has expected frame -> return it.
        - If stash has older -> discard.
        - If stash has future -> keep and wait for expected.
        - From queue: discard old (< expected), stash future (> expected), return expected.
        """
        expected = self.frame
        start = time.time()

        # 1) check stash first
        st = self._stash[idx]
        if st is not None:
            if st.frame == expected:
                self._stash[idx] = None
                return st
            elif st.frame < expected:
                self._stash[idx] = None  # discard old
            else:
                # st.frame > expected: keep it for future tick, continue waiting
                pass

        # 2) pull from queue until we get expected or timeout
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                raise queue.Empty(
                    f"[CarlaSyncMode] Timeout waiting for frame={expected} on queue#{idx}"
                )

            item = sensor_queue.get(timeout=min(remaining, 0.5))
            f = item.frame

            if f == expected:
                return item
            elif f < expected:
                # old frame -> discard and continue
                continue
            else:
                # future frame -> stash it and continue waiting for expected
                self._stash[idx] = item
                continue

    def close(self):
        """Restore world settings and stop sensors."""
        # Restore settings
        try:
            if self._settings is not None:
                self.world.apply_settings(self._settings)
        except Exception as e:
            print(f"[CarlaSyncMode] Warning: failed to restore settings: {e}")

        # Stop sensors
        for s in self.sensors:
            try:
                s.stop()
            except Exception as e:
                print(f"[CarlaSyncMode] Warning: failed to stop sensor: {e}")

        # Drain queues
        for q in self._queues:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

        self._queues.clear()
        self._stash.clear()
        self.sensors = ()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # don't suppress exceptions
        return False
