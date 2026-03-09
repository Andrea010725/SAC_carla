#!/usr/bin/env python3
from __future__ import annotations

"""
MVP test script:
Use semantic camera + LiDAR to estimate lane(6) and obstacle(K*3),
then compare against GT from CARLA map/world APIs and current SAC input slices.

Default scenario: cones (copied from current repo config flow).
"""

import argparse
import csv
import json
import math
import os
import queue
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

carla = None
CarlaEnv = None


SEM_ROAD_LINE = 6
SEM_ROAD = 7
SEM_SIDEWALK = 8
DRIVABLE_SEM = {SEM_ROAD_LINE, SEM_ROAD, SEM_SIDEWALK}

LANE_NAMES = [
    "lane_center_offset",
    "lane_heading_diff",
    "lane_width",
    "next_wp_dx",
    "next_wp_dy",
    "road_curvature",
]

OBS_NAMES = ["forward_dist", "lateral_dist", "relative_speed"]
SAC_LANE_NAMES = [
    "lane_dev_norm",
    "sin_heading_err",
    "cos_heading_err",
    "lane_width_norm",
    "wp_rel_x_norm",
    "wp_rel_y_norm",
]
SAC_OBS_NAMES = ["rel_x_norm", "rel_y_norm", "dist_norm"]


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _semantic_labels(sem_image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(sem_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((sem_image.height, sem_image.width, 4))
    for ch in (2, 1, 0):
        uniq = np.unique(arr[:, :, ch])
        if np.any(np.isin(uniq, [SEM_ROAD_LINE, SEM_ROAD, 10, 19, 20])):
            return arr[:, :, ch]
    return arr[:, :, 2]


def _lidar_points(lidar: carla.LidarMeasurement) -> np.ndarray:
    pts = np.frombuffer(lidar.raw_data, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return pts.reshape((-1, 4))


def _transform_points(points_xyz: np.ndarray, tf: carla.Transform) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mat = np.array(tf.get_matrix(), dtype=np.float32)  # 4x4
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([points_xyz, ones], axis=1)
    world_h = pts_h @ mat.T
    return world_h[:, :3]


def _project_world_to_camera(
    world_xyz: np.ndarray,
    cam_tf: carla.Transform,
    img_w: int,
    img_h: int,
    fov_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if world_xyz.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    w2c = np.array(cam_tf.get_inverse_matrix(), dtype=np.float32)
    ones = np.ones((world_xyz.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([world_xyz, ones], axis=1)
    cam_h = pts_h @ w2c.T

    # UE4 -> standard camera coords
    x = cam_h[:, 1]
    y = -cam_h[:, 2]
    z = cam_h[:, 0]

    f = img_w / (2.0 * math.tan(math.radians(float(fov_deg) / 2.0)))
    cx = img_w * 0.5
    cy = img_h * 0.5

    valid = z > 0.1
    u = np.zeros_like(z, dtype=np.float32)
    v = np.zeros_like(z, dtype=np.float32)
    u[valid] = (f * x[valid] / z[valid]) + cx
    v[valid] = (f * y[valid] / z[valid]) + cy
    valid = valid & (u >= 0.0) & (u < img_w) & (v >= 0.0) & (v < img_h)
    return u, v, valid


def _world_to_ego_xy(
    world_xyz: np.ndarray,
    ego_tf: carla.Transform,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if world_xyz.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()
    left_x = -float(right.x)
    left_y = -float(right.y)

    dx = world_xyz[:, 0] - float(ego_loc.x)
    dy = world_xyz[:, 1] - float(ego_loc.y)
    dz = world_xyz[:, 2] - float(ego_loc.z)

    rel_x = dx * float(fwd.x) + dy * float(fwd.y)
    rel_y = dx * left_x + dy * left_y
    return rel_x.astype(np.float32), rel_y.astype(np.float32), dz.astype(np.float32)


class SensorSuite:
    def __init__(
        self,
        world: carla.World,
        ego: carla.Actor,
        fixed_dt: float,
        sem_w: int = 640,
        sem_h: int = 360,
        sem_fov: float = 100.0,
        lidar_range: float = 60.0,
    ):
        self.world = world
        self.ego = ego
        self.sem_w = int(sem_w)
        self.sem_h = int(sem_h)
        self.sem_fov = float(sem_fov)
        self.sem_queue: "queue.Queue[carla.Image]" = queue.Queue()
        self.lidar_queue: "queue.Queue[carla.LidarMeasurement]" = queue.Queue()
        self.sem_stash: Optional[carla.Image] = None
        self.lidar_stash: Optional[carla.LidarMeasurement] = None
        self.sem_cam: Optional[carla.Sensor] = None
        self.lidar: Optional[carla.Sensor] = None
        self._actors: List[carla.Actor] = []

        bp_lib = world.get_blueprint_library()

        sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", str(self.sem_w))
        sem_bp.set_attribute("image_size_y", str(self.sem_h))
        sem_bp.set_attribute("fov", f"{self.sem_fov:.1f}")
        sem_tf = carla.Transform(carla.Location(x=1.60, z=1.70))
        self.sem_cam = world.try_spawn_actor(sem_bp, sem_tf, attach_to=ego)
        if self.sem_cam is None:
            raise RuntimeError("Failed to spawn semantic camera.")
        self.sem_cam.listen(self.sem_queue.put)
        self._actors.append(self.sem_cam)

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", f"{float(lidar_range):.1f}")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("points_per_second", "90000")
        lidar_bp.set_attribute("rotation_frequency", f"{1.0 / max(float(fixed_dt), 1e-3):.2f}")
        lidar_bp.set_attribute("upper_fov", "10.0")
        lidar_bp.set_attribute("lower_fov", "-30.0")
        lidar_tf = carla.Transform(carla.Location(x=0.00, z=1.90))
        self.lidar = world.try_spawn_actor(lidar_bp, lidar_tf, attach_to=ego)
        if self.lidar is None:
            raise RuntimeError("Failed to spawn LiDAR.")
        self.lidar.listen(self.lidar_queue.put)
        self._actors.append(self.lidar)

    def _wait_frame(
        self,
        q: queue.Queue,
        target_frame: int,
        timeout_s: float,
        stash_name: str,
    ):
        stash = getattr(self, stash_name)
        if stash is not None:
            if stash.frame == target_frame:
                setattr(self, stash_name, None)
                return stash
            if stash.frame > target_frame:
                return None
            setattr(self, stash_name, None)

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            remain = max(0.001, deadline - time.time())
            try:
                item = q.get(timeout=remain)
            except queue.Empty:
                return None
            if item.frame < target_frame:
                continue
            if item.frame == target_frame:
                return item
            setattr(self, stash_name, item)
            return None
        return None

    def poll(self, world_frame: int, timeout_s: float = 1.5):
        sem = self._wait_frame(self.sem_queue, world_frame, timeout_s, "sem_stash")
        lidar = self._wait_frame(self.lidar_queue, world_frame, timeout_s, "lidar_stash")
        return sem, lidar

    def close(self):
        for s in (self.sem_cam, self.lidar):
            if s is None:
                continue
            try:
                s.stop()
            except Exception:
                pass
        for a in reversed(self._actors):
            try:
                a.destroy()
            except Exception:
                pass
        self._actors.clear()
        self.sem_cam = None
        self.lidar = None


class SensorEstimator:
    def __init__(
        self,
        obstacle_k: int,
        obs_range: float,
        front_fwd_min: float,
        front_lat_tol: float,
    ):
        self.k = int(obstacle_k)
        self.obs_range = float(obs_range)
        self.front_fwd_min = float(front_fwd_min)
        self.front_lat_tol = float(front_lat_tol)
        self.prev_clusters: List[Tuple[float, float, float]] = []
        self.prev_ts: Optional[float] = None

    def reset(self):
        self.prev_clusters = []
        self.prev_ts = None

    def estimate(
        self,
        carla_env: CarlaEnv,
        sem: carla.Image,
        lidar: carla.LidarMeasurement,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points_lidar = _lidar_points(lidar)
        if points_lidar.shape[0] == 0:
            lane = np.zeros((6,), dtype=np.float32)
            obs = np.zeros((self.k, 3), dtype=np.float32)
            return lane, obs, canonical_lane_to_sac(lane), canonical_obs_to_sac(obs, self.obs_range)

        sem_labels = _semantic_labels(sem)
        lidar_local_xyz = points_lidar[:, :3]
        world_xyz = _transform_points(lidar_local_xyz, lidar.transform)
        u, v, valid = _project_world_to_camera(
            world_xyz=world_xyz,
            cam_tf=sem.transform,
            img_w=int(sem.width),
            img_h=int(sem.height),
            fov_deg=float(getattr(sem, "fov", 100.0)),
        )

        if not np.any(valid):
            lane = np.zeros((6,), dtype=np.float32)
            obs = np.zeros((self.k, 3), dtype=np.float32)
            return lane, obs, canonical_lane_to_sac(lane), canonical_obs_to_sac(obs, self.obs_range)

        idx = np.where(valid)[0]
        ui = u[idx].astype(np.int32)
        vi = v[idx].astype(np.int32)
        labels = sem_labels[vi, ui].astype(np.uint8)
        valid_world = world_xyz[idx]

        ego_tf = carla_env.ego.get_transform()
        rel_x, rel_y, rel_z = _world_to_ego_xy(valid_world, ego_tf)
        ego_points = np.stack([rel_x, rel_y, rel_z], axis=1)

        lane = self._estimate_lane_from_points(ego_points, labels)
        obs = self._estimate_obstacles_from_points(ego_points, labels, float(lidar.timestamp))
        lane_sac = canonical_lane_to_sac(lane)
        obs_sac = canonical_obs_to_sac(obs, self.obs_range)
        return lane, obs, lane_sac, obs_sac

    def _estimate_lane_from_points(self, ego_points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        x = ego_points[:, 0]
        y = ego_points[:, 1]
        z = ego_points[:, 2]
        base_mask = (x > 1.0) & (x < 30.0) & (np.abs(y) < 12.0) & (z > -3.0) & (z < 2.0)

        line_mask = base_mask & (labels == SEM_ROAD_LINE)
        road_mask = base_mask & np.isin(labels, [SEM_ROAD, SEM_ROAD_LINE])

        xs = []
        centers = []
        widths = []
        for center_x in np.arange(3.0, 24.1, 2.0):
            half = 0.9
            line_sel = line_mask & (x >= center_x - half) & (x <= center_x + half)
            road_sel = road_mask & (x >= center_x - half) & (x <= center_x + half)

            y_left = None
            y_right = None
            if np.count_nonzero(line_sel) >= 8:
                ys_line = y[line_sel]
                left_candidates = ys_line[ys_line > 0.2]
                right_candidates = ys_line[ys_line < -0.2]
                if left_candidates.size >= 3 and right_candidates.size >= 3:
                    y_left = float(np.percentile(left_candidates, 35))
                    y_right = float(np.percentile(right_candidates, 65))

            if (y_left is None or y_right is None) and np.count_nonzero(road_sel) >= 20:
                ys_road = y[road_sel]
                y_left = float(np.percentile(ys_road, 85))
                y_right = float(np.percentile(ys_road, 15))

            if y_left is None or y_right is None:
                continue

            width = y_left - y_right
            if width < 1.8 or width > 8.5:
                continue

            xs.append(float(center_x))
            centers.append(float(0.5 * (y_left + y_right)))
            widths.append(float(width))

        if len(xs) < 2:
            return np.zeros((6,), dtype=np.float32)

        x_arr = np.asarray(xs, dtype=np.float32)
        y_arr = np.asarray(centers, dtype=np.float32)
        if len(xs) >= 3:
            poly = np.polyfit(x_arr, y_arr, 2)
            a, b, c = float(poly[0]), float(poly[1]), float(poly[2])
        else:
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            a, b, c = 0.0, float(slope), float(intercept)

        lookahead = 12.0
        lane_center_at_ego = c
        lane_center_offset = float(-lane_center_at_ego)
        slope_look = float(2.0 * a * lookahead + b)
        lane_heading_diff = float(math.atan(slope_look))
        lane_width = float(np.median(np.asarray(widths, dtype=np.float32))) if widths else 3.5
        next_wp_dx = float(lookahead)
        next_wp_dy = float(a * lookahead * lookahead + b * lookahead + c)
        road_curvature = float((2.0 * a) / max(1e-6, (1.0 + slope_look * slope_look) ** 1.5))

        return np.array(
            [
                lane_center_offset,
                lane_heading_diff,
                lane_width,
                next_wp_dx,
                next_wp_dy,
                road_curvature,
            ],
            dtype=np.float32,
        )

    def _cluster_points_grid(self, pts: np.ndarray) -> List[np.ndarray]:
        if pts.shape[0] == 0:
            return []
        cell_size = 0.55
        min_points = 8
        cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i in range(pts.shape[0]):
            cx = int(math.floor(float(pts[i, 0]) / cell_size))
            cy = int(math.floor(float(pts[i, 1]) / cell_size))
            cells[(cx, cy)].append(i)

        visited = set()
        clusters: List[np.ndarray] = []
        for key in cells.keys():
            if key in visited:
                continue
            stack = [key]
            comp_keys = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp_keys.append(cur)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nxt = (cur[0] + dx, cur[1] + dy)
                        if nxt in cells and nxt not in visited:
                            stack.append(nxt)

            idxs: List[int] = []
            for ck in comp_keys:
                idxs.extend(cells[ck])
            if len(idxs) >= min_points:
                clusters.append(np.asarray(idxs, dtype=np.int32))
        return clusters

    def _estimate_obstacles_from_points(
        self,
        ego_points: np.ndarray,
        labels: np.ndarray,
        ts: float,
    ) -> np.ndarray:
        x = ego_points[:, 0]
        y = ego_points[:, 1]
        z = ego_points[:, 2]
        mask = (
            (x > 1.2)
            & (x < self.obs_range)
            & (np.abs(y) < self.obs_range * 0.8)
            & (z > -1.3)
            & (z < 2.8)
            & (~np.isin(labels, list(DRIVABLE_SEM)))
        )
        candidate = ego_points[mask]
        if candidate.shape[0] == 0:
            self.prev_clusters = []
            self.prev_ts = ts
            return np.zeros((self.k, 3), dtype=np.float32)

        clusters_idx = self._cluster_points_grid(candidate[:, :2])
        if not clusters_idx:
            self.prev_clusters = []
            self.prev_ts = ts
            return np.zeros((self.k, 3), dtype=np.float32)

        clusters = []
        for idxs in clusters_idx:
            pts = candidate[idxs]
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            cz = float(np.mean(pts[:, 2]))
            if cx < -2.0:
                continue
            dist = float(math.hypot(cx, cy))
            clusters.append((cx, cy, cz, dist))

        dt = None if self.prev_ts is None else max(1e-3, float(ts - self.prev_ts))
        out_entries = []
        used_prev = set()
        for cx, cy, _cz, dist in clusters:
            rel_speed = 0.0
            if dt is not None and self.prev_clusters:
                best_j = -1
                best_d = 1e9
                for j, (px, py, _pts) in enumerate(self.prev_clusters):
                    if j in used_prev:
                        continue
                    d = (cx - px) * (cx - px) + (cy - py) * (cy - py)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0 and best_d <= (2.5 * 2.5):
                    used_prev.add(best_j)
                    px, _py, _pts = self.prev_clusters[best_j]
                    rel_speed = float(-(cx - px) / dt)
            out_entries.append((cx, cy, rel_speed, dist))

        self.prev_clusters = [(e[0], e[1], ts) for e in out_entries]
        self.prev_ts = ts

        selected = sort_obstacles_by_priority(
            entries=[(e[0], e[1], e[2]) for e in out_entries],
            front_fwd_min=self.front_fwd_min,
            front_lat_tol=self.front_lat_tol,
            top_k=self.k,
        )
        return selected


def sort_obstacles_by_priority(
    entries: List[Tuple[float, float, float]],
    front_fwd_min: float,
    front_lat_tol: float,
    top_k: int,
) -> np.ndarray:
    scored = []
    for fwd, lat, rel_v in entries:
        dist = float(math.hypot(fwd, lat))
        is_front = bool(fwd > front_fwd_min)
        is_front_lane = bool(is_front and (abs(lat) <= front_lat_tol))
        prio = 0 if is_front_lane else (1 if is_front else 2)
        scored.append((prio, dist, float(fwd), float(lat), float(rel_v)))
    scored.sort(key=lambda x: (x[0], x[1]))
    picked = scored[: int(top_k)]
    out = np.zeros((int(top_k), 3), dtype=np.float32)
    for i, (_p, _d, fwd, lat, rel_v) in enumerate(picked):
        out[i, 0] = fwd
        out[i, 1] = lat
        out[i, 2] = rel_v
    return out


def canonical_lane_to_sac(lane6: np.ndarray) -> np.ndarray:
    lane_center_offset, lane_heading_diff, lane_width, next_wp_dx, next_wp_dy, _curv = [float(x) for x in lane6]
    lane_dev_norm = float(np.clip(abs(lane_center_offset) / max(1e-3, lane_width), 0.0, 3.0))
    lane_width_norm = float(np.clip(lane_width / 4.0, 0.0, 2.0))
    wp_rel_x_norm = float(np.clip(next_wp_dx / 20.0, -2.0, 2.0))
    wp_rel_y_norm = float(np.clip(next_wp_dy / 10.0, -2.0, 2.0))
    return np.array(
        [
            lane_dev_norm,
            math.sin(lane_heading_diff),
            math.cos(lane_heading_diff),
            lane_width_norm,
            wp_rel_x_norm,
            wp_rel_y_norm,
        ],
        dtype=np.float32,
    )


def canonical_obs_to_sac(obs_k3: np.ndarray, obs_range: float) -> np.ndarray:
    out = np.zeros_like(obs_k3, dtype=np.float32)
    for i in range(obs_k3.shape[0]):
        fwd = float(obs_k3[i, 0])
        lat = float(obs_k3[i, 1])
        out[i, 0] = float(np.clip(fwd / max(1e-3, obs_range), -1.0, 1.0))
        out[i, 1] = float(np.clip(lat / max(1e-3, obs_range), -1.0, 1.0))
        out[i, 2] = float(np.clip(math.hypot(fwd, lat) / max(1e-3, obs_range), 0.0, 1.0))
    return out


def compute_lane_gt(carla_env: CarlaEnv) -> np.ndarray:
    ego = carla_env.ego
    amap = carla_env.map
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    wp = amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return np.zeros((6,), dtype=np.float32)

    lane_tf = wp.transform
    lane_loc = lane_tf.location
    lane_fwd = lane_tf.get_forward_vector()
    lane_left_x = -float(lane_fwd.y)
    lane_left_y = float(lane_fwd.x)

    dx = float(ego_loc.x - lane_loc.x)
    dy = float(ego_loc.y - lane_loc.y)
    lane_center_offset = float(dx * lane_left_x + dy * lane_left_y)

    lane_yaw = math.atan2(float(lane_fwd.y), float(lane_fwd.x))
    ego_yaw = math.radians(float(ego_tf.rotation.yaw))
    lane_heading_diff = float(wrap_angle(lane_yaw - ego_yaw))

    lane_width = float(getattr(wp, "lane_width", 3.5))

    target_wp = getattr(carla_env, "target_wp", None)
    if target_wp is None:
        nxt = wp.next(float(getattr(carla_env, "wp_step_dist", 5.0)))
        target_wp = nxt[0] if nxt else wp
    tgt = target_wp.transform.location

    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()
    ego_left_x = -float(ego_right.x)
    ego_left_y = -float(ego_right.y)

    tdx = float(tgt.x - ego_loc.x)
    tdy = float(tgt.y - ego_loc.y)
    next_wp_dx = float(tdx * float(ego_fwd.x) + tdy * float(ego_fwd.y))
    next_wp_dy = float(tdx * ego_left_x + tdy * ego_left_y)

    curv = 0.0
    step = 2.0
    prevs = wp.previous(step)
    nexts = wp.next(step)
    if prevs and nexts:
        yaw_p = math.radians(float(prevs[0].transform.rotation.yaw))
        yaw_n = math.radians(float(nexts[0].transform.rotation.yaw))
        dyaw = wrap_angle(yaw_n - yaw_p)
        curv = float(dyaw / (2.0 * step))

    return np.array(
        [
            lane_center_offset,
            lane_heading_diff,
            lane_width,
            next_wp_dx,
            next_wp_dy,
            curv,
        ],
        dtype=np.float32,
    )


def _collect_world_obstacles(world: carla.World, ego: carla.Actor) -> List[carla.Actor]:
    actors = world.get_actors()
    out: List[carla.Actor] = []
    seen = set()

    def _append(a: carla.Actor):
        aid = int(a.id)
        if aid == int(ego.id):
            return
        if aid in seen:
            return
        seen.add(aid)
        out.append(a)

    for v in actors.filter("vehicle.*"):
        _append(v)
    for w in actors.filter("walker.pedestrian.*"):
        _append(w)
    for p in actors.filter("static.prop.*"):
        tid = getattr(p, "type_id", "").lower()
        if any(k in tid for k in ["cone", "trafficcone", "barrier", "construction", "warning", "box"]):
            _append(p)
    return out


def compute_obstacle_gt(
    carla_env: CarlaEnv,
    obstacle_k: int,
    obs_range: float,
    front_fwd_min: float,
    front_lat_tol: float,
) -> np.ndarray:
    ego = carla_env.ego
    world = carla_env.world
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_vel = ego.get_velocity()
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()
    ego_left_x = -float(ego_right.x)
    ego_left_y = -float(ego_right.y)

    entries: List[Tuple[float, float, float]] = []
    for a in _collect_world_obstacles(world, ego):
        try:
            loc = a.get_location()
        except Exception:
            continue
        dx = float(loc.x - ego_loc.x)
        dy = float(loc.y - ego_loc.y)

        fwd = float(dx * float(ego_fwd.x) + dy * float(ego_fwd.y))
        lat = float(dx * ego_left_x + dy * ego_left_y)
        dist = float(math.hypot(fwd, lat))
        if dist > float(obs_range):
            continue

        rel_speed = 0.0
        try:
            av = a.get_velocity()
            rel_speed = float(
                (float(ego_vel.x) - float(av.x)) * float(ego_fwd.x)
                + (float(ego_vel.y) - float(av.y)) * float(ego_fwd.y)
            )
        except Exception:
            rel_speed = float(float(ego_vel.x) * float(ego_fwd.x) + float(ego_vel.y) * float(ego_fwd.y))

        entries.append((fwd, lat, rel_speed))

    return sort_obstacles_by_priority(
        entries=entries,
        front_fwd_min=float(front_fwd_min),
        front_lat_tol=float(front_lat_tol),
        top_k=int(obstacle_k),
    )


def build_cfg(args: argparse.Namespace):
    cfg = SimpleNamespace()
    cfg.map_name = str(args.map)
    cfg.render = bool(args.render)
    cfg.spectator_mode = str(args.spectator_mode)

    cfg.random_scenario = False
    cfg.scenario = str(args.scenario)
    cfg.scenario_pool = [str(args.scenario)]
    cfg.scenario_weights = {str(args.scenario): 1.0}

    cfg.observations_type = "state_lane_obstacles"
    cfg.obs_obstacle_k = int(args.obstacle_k)
    cfg.obs_obstacle_range = float(args.obs_range)
    cfg.max_episode_steps = int(max(args.episode_max_steps, args.steps + args.warmup_steps + 20))

    cfg.use_yref_in_steer = False
    cfg.use_yref_mapping = False
    cfg.yref_steer_gain = 0.0
    cfg.yref_gain = 0.0
    cfg.yref_penalty = 0.0
    cfg.enable_debug_drawing = False
    cfg.debug_obstacle_obs = False
    cfg.debug_obstacle_gate = False
    cfg.seed = int(args.seed)
    return cfg


def build_csv_fields(obstacle_k: int) -> List[str]:
    fields = [
        "episode",
        "env_step",
        "frame",
        "reward",
        "done",
    ]
    for n in LANE_NAMES:
        fields.extend([f"lane_gt_{n}", f"lane_pred_{n}", f"lane_abs_{n}"])
    for n in SAC_LANE_NAMES:
        fields.extend([f"sac_lane_gt_{n}", f"sac_lane_pred_{n}", f"sac_lane_abs_{n}"])
    for i in range(int(obstacle_k)):
        for n in OBS_NAMES:
            fields.extend([f"obs_gt_{i}_{n}", f"obs_pred_{i}_{n}", f"obs_abs_{i}_{n}"])
    for i in range(int(obstacle_k)):
        for n in SAC_OBS_NAMES:
            fields.extend([f"sac_obs_gt_{i}_{n}", f"sac_obs_pred_{i}_{n}", f"sac_obs_abs_{i}_{n}"])
    return fields


def action_for_step(args: argparse.Namespace, global_step: int) -> np.ndarray:
    throttle = float(np.clip(args.throttle, -1.0, 1.0))
    mode = str(args.steer_mode).lower()
    if mode == "constant":
        steer = 0.0
    elif mode == "sine":
        steer = float(np.clip(args.steer_amplitude * math.sin(0.06 * global_step), -1.0, 1.0))
    else:
        steer = float(np.random.uniform(-abs(args.steer_amplitude), abs(args.steer_amplitude)))
    return np.array([throttle, steer, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVP test: semantic camera + LiDAR estimate lane/obstacle, compare with GT and SAC obs slices."
    )
    parser.add_argument("--steps", type=int, default=120, help="Number of valid comparison steps to record.")
    parser.add_argument("--episode-max-steps", type=int, default=320)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--map", type=str, default="Town05")
    parser.add_argument("--scenario", type=str, default="cones")
    parser.add_argument("--render", type=int, choices=[0, 1], default=1, help="1 keeps rendering on (needed for camera).")
    parser.add_argument("--spectator-mode", type=str, default="none")

    parser.add_argument("--obstacle-k", type=int, default=5)
    parser.add_argument("--obs-range", type=float, default=50.0)
    parser.add_argument("--front-fwd-min", type=float, default=-0.5)
    parser.add_argument("--front-lat-tol", type=float, default=2.8)

    parser.add_argument("--sem-width", type=int, default=640)
    parser.add_argument("--sem-height", type=int, default=360)
    parser.add_argument("--sem-fov", type=float, default=100.0)
    parser.add_argument("--lidar-range", type=float, default=60.0)
    parser.add_argument("--sensor-timeout", type=float, default=1.5)

    parser.add_argument("--throttle", type=float, default=0.35)
    parser.add_argument("--steer-mode", type=str, choices=["constant", "sine", "random"], default="sine")
    parser.add_argument("--steer-amplitude", type=float, default=0.12)

    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--max-total-env-steps", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    global carla, CarlaEnv
    import carla as carla_mod
    from carla_base.carla_env import CarlaEnv as CarlaEnvCls

    carla = carla_mod
    CarlaEnv = CarlaEnvCls

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_csv = os.path.join("result", f"sensor_mvp_cones_compare_{ts}.csv")
    default_json = os.path.join("result", f"sensor_mvp_cones_summary_{ts}.json")
    output_csv = args.output_csv.strip() or default_csv
    output_json = args.output_json.strip() or default_json
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    cfg = build_cfg(args)
    env: Optional[CarlaEnv] = None
    suite: Optional[SensorSuite] = None

    lane_errs = []
    sac_lane_errs = []
    obs_errs = []
    sac_obs_errs = []

    valid_steps = 0
    total_env_steps = 0
    missing_sensor_frames = 0
    episode_idx = 0

    csv_fields = build_csv_fields(int(args.obstacle_k))
    estimator = SensorEstimator(
        obstacle_k=int(args.obstacle_k),
        obs_range=float(args.obs_range),
        front_fwd_min=float(args.front_fwd_min),
        front_lat_tol=float(args.front_lat_tol),
    )

    print("[MVP] start")
    print(f"[MVP] scenario={args.scenario} map={args.map} steps={args.steps} K={args.obstacle_k}")
    print(f"[MVP] output_csv={output_csv}")
    print(f"[MVP] output_json={output_json}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        try:
            env = CarlaEnv(cfg, carla_port=int(args.carla_port), tm_port=int(args.tm_port))
            while valid_steps < int(args.steps) and total_env_steps < int(args.max_total_env_steps):
                episode_idx += 1
                obs = env.reset()
                estimator.reset()

                if suite is not None:
                    suite.close()
                suite = SensorSuite(
                    world=env.world,
                    ego=env.ego,
                    fixed_dt=float(env.fixed_dt),
                    sem_w=int(args.sem_width),
                    sem_h=int(args.sem_height),
                    sem_fov=float(args.sem_fov),
                    lidar_range=float(args.lidar_range),
                )

                # sensor warmup
                warmup_done = False
                for _ in range(int(args.warmup_steps)):
                    _obs_w, _rew_w, done_w, _info_w = env.step(np.array([0.15, 0.0, 0.0], dtype=np.float32))
                    total_env_steps += 1
                    if done_w:
                        warmup_done = True
                        break
                if warmup_done:
                    continue

                done = False
                env_step_in_ep = 0
                while (not done) and valid_steps < int(args.steps) and total_env_steps < int(args.max_total_env_steps):
                    action = action_for_step(args, total_env_steps)
                    next_obs, reward, done, _info = env.step(action)
                    total_env_steps += 1
                    env_step_in_ep += 1

                    frame = int(env.world.get_snapshot().frame)
                    sem, lidar = suite.poll(frame, timeout_s=float(args.sensor_timeout))
                    if sem is None or lidar is None:
                        missing_sensor_frames += 1
                        continue

                    lane_pred, obs_pred_k3, lane_sac_pred, obs_sac_pred_k3 = estimator.estimate(
                        carla_env=env,
                        sem=sem,
                        lidar=lidar,
                    )

                    lane_gt = compute_lane_gt(env)
                    obs_gt_k3 = compute_obstacle_gt(
                        carla_env=env,
                        obstacle_k=int(args.obstacle_k),
                        obs_range=float(args.obs_range),
                        front_fwd_min=float(args.front_fwd_min),
                        front_lat_tol=float(args.front_lat_tol),
                    )

                    base_dim = int(getattr(env, "base_state_dim", 9))
                    lane_dim = int(getattr(env, "lane_dim", 6))
                    lane_sac_gt = np.asarray(next_obs[base_dim : base_dim + lane_dim], dtype=np.float32)
                    obs_sac_gt_flat = np.asarray(
                        next_obs[base_dim + lane_dim : base_dim + lane_dim + int(args.obstacle_k) * 3],
                        dtype=np.float32,
                    )
                    obs_sac_gt_k3 = obs_sac_gt_flat.reshape((int(args.obstacle_k), 3))

                    lane_abs = np.abs(lane_pred - lane_gt)
                    obs_abs = np.abs(obs_pred_k3 - obs_gt_k3)
                    lane_sac_abs = np.abs(lane_sac_pred - lane_sac_gt)
                    obs_sac_abs = np.abs(obs_sac_pred_k3 - obs_sac_gt_k3)

                    lane_errs.append(lane_abs)
                    obs_errs.append(obs_abs)
                    sac_lane_errs.append(lane_sac_abs)
                    sac_obs_errs.append(obs_sac_abs)

                    row = {
                        "episode": int(episode_idx),
                        "env_step": int(env_step_in_ep),
                        "frame": int(frame),
                        "reward": float(reward),
                        "done": int(done),
                    }
                    for i, n in enumerate(LANE_NAMES):
                        row[f"lane_gt_{n}"] = float(lane_gt[i])
                        row[f"lane_pred_{n}"] = float(lane_pred[i])
                        row[f"lane_abs_{n}"] = float(lane_abs[i])

                    for i, n in enumerate(SAC_LANE_NAMES):
                        row[f"sac_lane_gt_{n}"] = float(lane_sac_gt[i])
                        row[f"sac_lane_pred_{n}"] = float(lane_sac_pred[i])
                        row[f"sac_lane_abs_{n}"] = float(lane_sac_abs[i])

                    for i in range(int(args.obstacle_k)):
                        for j, n in enumerate(OBS_NAMES):
                            row[f"obs_gt_{i}_{n}"] = float(obs_gt_k3[i, j])
                            row[f"obs_pred_{i}_{n}"] = float(obs_pred_k3[i, j])
                            row[f"obs_abs_{i}_{n}"] = float(obs_abs[i, j])

                    for i in range(int(args.obstacle_k)):
                        for j, n in enumerate(SAC_OBS_NAMES):
                            row[f"sac_obs_gt_{i}_{n}"] = float(obs_sac_gt_k3[i, j])
                            row[f"sac_obs_pred_{i}_{n}"] = float(obs_sac_pred_k3[i, j])
                            row[f"sac_obs_abs_{i}_{n}"] = float(obs_sac_abs[i, j])

                    writer.writerow(row)
                    valid_steps += 1

                    if valid_steps % 20 == 0:
                        print(
                            f"[MVP] valid_steps={valid_steps}/{args.steps} "
                            f"miss={missing_sensor_frames} total_env_steps={total_env_steps}"
                        )
        finally:
            if suite is not None:
                suite.close()
            if env is not None:
                env.close()

    if valid_steps == 0:
        print("[MVP] No valid samples collected.")
        summary = {
            "valid_steps": 0,
            "total_env_steps": int(total_env_steps),
            "missing_sensor_frames": int(missing_sensor_frames),
            "note": "No synchronized semantic/lidar frames were collected.",
        }
        with open(output_json, "w") as jf:
            json.dump(summary, jf, indent=2)
        return

    lane_mae = np.mean(np.stack(lane_errs, axis=0), axis=0)
    obs_mae = np.mean(np.stack(obs_errs, axis=0), axis=(0, 1))
    sac_lane_mae = np.mean(np.stack(sac_lane_errs, axis=0), axis=0)
    sac_obs_mae = np.mean(np.stack(sac_obs_errs, axis=0), axis=(0, 1))

    summary = {
        "scenario": str(args.scenario),
        "map": str(args.map),
        "valid_steps": int(valid_steps),
        "total_env_steps": int(total_env_steps),
        "missing_sensor_frames": int(missing_sensor_frames),
        "lane_mae": {k: float(v) for k, v in zip(LANE_NAMES, lane_mae.tolist())},
        "obstacle_mae": {k: float(v) for k, v in zip(OBS_NAMES, obs_mae.tolist())},
        "sac_lane_mae": {k: float(v) for k, v in zip(SAC_LANE_NAMES, sac_lane_mae.tolist())},
        "sac_obstacle_mae": {k: float(v) for k, v in zip(SAC_OBS_NAMES, sac_obs_mae.tolist())},
        "output_csv": os.path.abspath(output_csv),
    }
    with open(output_json, "w") as jf:
        json.dump(summary, jf, indent=2)

    print("\n[MVP] summary")
    print(json.dumps(summary, indent=2))
    print(f"[MVP] wrote: {os.path.abspath(output_csv)}")
    print(f"[MVP] wrote: {os.path.abspath(output_json)}")


if __name__ == "__main__":
    main()
