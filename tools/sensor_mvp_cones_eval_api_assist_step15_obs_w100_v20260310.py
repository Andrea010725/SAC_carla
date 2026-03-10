#!/usr/bin/env python3
from __future__ import annotations

"""
MVP test script:
Use semantic camera + LiDAR to estimate lane(6) and obstacle(K*3),
then compare against GT from CARLA map/world APIs.

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
sys.path.append("/home/zhiwen/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
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
SCRIPT_VERSION = "sensor_mvp_api_assist_step15_obs_w100_v20260310"
LANE_LOOKAHEAD_FIXED_M = 12.0
LANE_DIR_MOTION_MIN_STEP_M = 0.10
LANE_DIR_OPPOSITE_MOTION = True
LANE_WIDTH_VALID_MIN_M = 2.0
LANE_WIDTH_VALID_MAX_M = 5.0
LANE_WIDTH_ROAD_CLAMP_MIN_M = 2.8
LANE_WIDTH_ROAD_CLAMP_MAX_M = 4.2
LANE_CENTER_SMOOTH_ALPHA = 0.75
LANE_WIDTH_SMOOTH_ALPHA = 0.70
LANE_LINE_NEIGHBOR_RADIUS_PX = 1
LANE_LINE_SLICE_MIN_POINTS = 4
LANE_LINE_SIDE_MIN_POINTS = 1
SEM_LIDAR_MIN_POINTS = 24
RADAR_MATCH_RADIUS_M = 3.0
USE_RADAR_FOR_OBS_SPEED = False
USE_API_LANE_ASSIST = True
USE_API_OBS_ASSIST = True
API_LANE_ASSIST_WEIGHT = 0.95
API_OBS_ASSIST_WEIGHT = 1.0


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def blend_angle(sensor_a: float, prior_a: float, prior_w: float) -> float:
    w = float(np.clip(prior_w, 0.0, 1.0))
    s = (1.0 - w) * math.sin(sensor_a) + w * math.sin(prior_a)
    c = (1.0 - w) * math.cos(sensor_a) + w * math.cos(prior_a)
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return float(sensor_a)
    return float(math.atan2(s, c))


def _semantic_labels(sem_image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(sem_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((sem_image.height, sem_image.width, 4))
    for ch in (2, 1, 0):
        uniq = np.unique(arr[:, :, ch])
        if np.any(np.isin(uniq, [SEM_ROAD_LINE, SEM_ROAD, 10, 19, 20])):
            return arr[:, :, ch]
    return arr[:, :, 2]


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    h, w = mask.shape
    pad = int(radius)
    padded = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dy in range(-pad, pad + 1):
        y0 = pad + dy
        y1 = y0 + h
        for dx in range(-pad, pad + 1):
            x0 = pad + dx
            x1 = x0 + w
            out |= padded[y0:y1, x0:x1]
    return out


def _lidar_points(lidar: carla.LidarMeasurement) -> np.ndarray:
    pts = np.frombuffer(lidar.raw_data, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return pts.reshape((-1, 4))


def _semantic_lidar_points_and_tags(sem_lidar) -> Tuple[np.ndarray, np.ndarray]:
    if sem_lidar is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    pts: List[Tuple[float, float, float]] = []
    tags: List[int] = []
    try:
        for det in sem_lidar:
            p = det.point
            pts.append((float(p.x), float(p.y), float(p.z)))
            tags.append(int(det.object_tag))
    except Exception:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    if not pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
    return np.asarray(pts, dtype=np.float32), np.asarray(tags, dtype=np.uint8)


def _radar_points_ego(radar_meas) -> np.ndarray:
    if radar_meas is None:
        return np.zeros((0, 3), dtype=np.float32)
    out: List[Tuple[float, float, float]] = []
    try:
        for det in radar_meas:
            depth = float(det.depth)
            az = float(det.azimuth)
            alt = float(det.altitude)
            # CARLA radar uses sensor-forward coordinates:
            # x=forward, y=right, z=up. Convert to ego-left convention for lateral.
            cos_alt = math.cos(alt)
            fwd = float(depth * cos_alt * math.cos(az))
            lat = float(-depth * cos_alt * math.sin(az))
            rel_speed = float(det.velocity)
            out.append((fwd, lat, rel_speed))
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if world_xyz.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            "empty",
        )
    w2c = np.array(cam_tf.get_inverse_matrix(), dtype=np.float32)
    ones = np.ones((world_xyz.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([world_xyz, ones], axis=1)
    cam_h = pts_h @ w2c.T

    f = img_w / (2.0 * math.tan(math.radians(float(fov_deg) / 2.0)))
    cx = img_w * 0.5
    cy = img_h * 0.5
    x0 = cam_h[:, 0]
    y0 = cam_h[:, 1]
    z0 = cam_h[:, 2]

    # Try several axis conventions and pick the one with max in-frame points.
    candidates = [
        ("ue_std", y0, -z0, x0),
        ("ue_flip_y", -y0, -z0, x0),
        ("direct_xyz", x0, y0, z0),
        ("direct_xy_negz", x0, y0, -z0),
    ]
    best = None
    best_count = -1
    for name, x, y, z in candidates:
        valid = z > 0.1
        u = np.zeros_like(z, dtype=np.float32)
        v = np.zeros_like(z, dtype=np.float32)
        u[valid] = (f * x[valid] / z[valid]) + cx
        v[valid] = (f * y[valid] / z[valid]) + cy
        valid = valid & (u >= 0.0) & (u < img_w) & (v >= 0.0) & (v < img_h)
        cnt = int(np.count_nonzero(valid))
        if cnt > best_count:
            best_count = cnt
            best = (u, v, valid, name)

    if best is None:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            "none",
        )
    return best


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
        radar_range: float = 60.0,
    ):
        self.world = world
        self.ego = ego
        self.sem_w = int(sem_w)
        self.sem_h = int(sem_h)
        self.sem_fov = float(sem_fov)
        self.sem_queue: "queue.Queue[carla.Image]" = queue.Queue()
        self.lidar_queue: "queue.Queue[carla.LidarMeasurement]" = queue.Queue()
        self.sem_lidar_queue: "queue.Queue" = queue.Queue()
        self.radar_queue: "queue.Queue" = queue.Queue()
        self.sem_stash: Optional[carla.Image] = None
        self.lidar_stash: Optional[carla.LidarMeasurement] = None
        self.sem_lidar_stash = None
        self.radar_stash = None
        self.sem_cam: Optional[carla.Sensor] = None
        self.lidar: Optional[carla.Sensor] = None
        self.sem_lidar: Optional[carla.Sensor] = None
        self.radar: Optional[carla.Sensor] = None
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

        sem_lidar_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
        sem_lidar_bp.set_attribute("range", f"{float(lidar_range):.1f}")
        sem_lidar_bp.set_attribute("channels", "32")
        sem_lidar_bp.set_attribute("points_per_second", "90000")
        sem_lidar_bp.set_attribute("rotation_frequency", f"{1.0 / max(float(fixed_dt), 1e-3):.2f}")
        sem_lidar_bp.set_attribute("upper_fov", "10.0")
        sem_lidar_bp.set_attribute("lower_fov", "-30.0")
        sem_lidar_tf = carla.Transform(carla.Location(x=0.00, z=1.90))
        self.sem_lidar = world.try_spawn_actor(sem_lidar_bp, sem_lidar_tf, attach_to=ego)
        if self.sem_lidar is None:
            raise RuntimeError("Failed to spawn semantic LiDAR.")
        self.sem_lidar.listen(self.sem_lidar_queue.put)
        self._actors.append(self.sem_lidar)

        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", "35")
        radar_bp.set_attribute("vertical_fov", "20")
        radar_bp.set_attribute("range", f"{float(radar_range):.1f}")
        radar_bp.set_attribute("points_per_second", "1500")
        radar_tf = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = world.try_spawn_actor(radar_bp, radar_tf, attach_to=ego)
        if self.radar is None:
            raise RuntimeError("Failed to spawn radar.")
        self.radar.listen(self.radar_queue.put)
        self._actors.append(self.radar)

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
        sem_lidar = self._wait_frame(self.sem_lidar_queue, world_frame, timeout_s, "sem_lidar_stash")
        radar = self._wait_frame(self.radar_queue, world_frame, timeout_s, "radar_stash")
        return sem, lidar, sem_lidar, radar

    def close(self):
        for s in (self.sem_cam, self.lidar, self.sem_lidar, self.radar):
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
        self.sem_lidar = None
        self.radar = None


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
        self.prev_ego_xy: Optional[Tuple[float, float]] = None
        self.prev_lane_dir_sign: float = -1.0 if LANE_DIR_OPPOSITE_MOTION else 1.0
        self.prev_lane_center_offset: Optional[float] = None
        self.prev_lane_width: Optional[float] = None

    def reset(self):
        self.prev_clusters = []
        self.prev_ts = None
        self.prev_ego_xy = None
        self.prev_lane_dir_sign = -1.0 if LANE_DIR_OPPOSITE_MOTION else 1.0
        self.prev_lane_center_offset = None
        self.prev_lane_width = None

    def estimate(
        self,
        carla_env: CarlaEnv,
        sem: carla.Image,
        lidar: carla.LidarMeasurement,
        sem_lidar=None,
        radar=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        points_lidar = _lidar_points(lidar) if lidar is not None else np.zeros((0, 4), dtype=np.float32)
        sem_lidar_local_xyz, sem_lidar_tags = _semantic_lidar_points_and_tags(sem_lidar)
        use_sem_lidar = sem_lidar_local_xyz.shape[0] >= int(SEM_LIDAR_MIN_POINTS)
        if points_lidar.shape[0] == 0 and sem_lidar_local_xyz.shape[0] == 0:
            lane = np.zeros((6,), dtype=np.float32)
            obs = np.zeros((self.k, 3), dtype=np.float32)
            meta = {
                "lidar_points": 0.0,
                "projected_points": 0.0,
                "projection_valid_ratio": 0.0,
                "used_lidar_only_fallback": 1.0,
                "lane_nonzero": 0.0,
                "obs_nonzero": 0.0,
                "projection_mode": "no_lidar",
                "lane_source_sem_lidar": 0.0,
                "obs_radar_matches": 0.0,
                "obs_candidate_clusters": 0.0,
                "obs_radar_points": 0.0,
            }
            return lane, obs, canonical_lane_to_sac(lane), canonical_obs_to_sac(obs, self.obs_range), meta

        used_lidar_only_fallback = 0.0
        if use_sem_lidar:
            sem_lidar_world = _transform_points(sem_lidar_local_xyz, sem_lidar.transform)
            valid_world = sem_lidar_world
            lane_labels = sem_lidar_tags.astype(np.uint8)
            obs_labels = sem_lidar_tags.astype(np.uint8)
            total_pts = int(sem_lidar_local_xyz.shape[0])
            valid_cnt = int(sem_lidar_local_xyz.shape[0])
            valid_ratio = 1.0
            proj_mode = "semantic_lidar"
        else:
            sem_labels = _semantic_labels(sem)
            lidar_local_xyz = points_lidar[:, :3]
            world_xyz = _transform_points(lidar_local_xyz, lidar.transform)
            u, v, valid, proj_mode = _project_world_to_camera(
                world_xyz=world_xyz,
                cam_tf=sem.transform,
                img_w=int(sem.width),
                img_h=int(sem.height),
                fov_deg=float(getattr(sem, "fov", 100.0)),
            )

            total_pts = int(points_lidar.shape[0])
            valid_cnt = int(np.count_nonzero(valid))
            valid_ratio = float(valid_cnt / max(1, total_pts))

            if valid_cnt <= 16:
                # projection is effectively unusable; fallback to lidar-only geometry
                used_lidar_only_fallback = 1.0
                valid_world = world_xyz
                lane_labels = np.full((valid_world.shape[0],), SEM_ROAD, dtype=np.uint8)
                obs_labels = np.zeros((valid_world.shape[0],), dtype=np.uint8)
            else:
                idx = np.where(valid)[0]
                ui = u[idx].astype(np.int32)
                vi = v[idx].astype(np.int32)
                labels = sem_labels[vi, ui].astype(np.uint8)
                valid_world = world_xyz[idx]
                if int(LANE_LINE_NEIGHBOR_RADIUS_PX) > 0:
                    line_mask = sem_labels == SEM_ROAD_LINE
                    line_hit_mask = _dilate_binary(line_mask, int(LANE_LINE_NEIGHBOR_RADIUS_PX))
                    lane_labels = labels.copy()
                    lane_labels[line_hit_mask[vi, ui]] = np.uint8(SEM_ROAD_LINE)
                else:
                    lane_labels = labels
                obs_labels = labels

        ego_tf = carla_env.ego.get_transform()
        ego_loc = ego_tf.location
        rel_x, rel_y, rel_z = _world_to_ego_xy(valid_world, ego_tf)
        ego_points = np.stack([rel_x, rel_y, rel_z], axis=1)

        motion_dir_xy = None
        if self.prev_ego_xy is not None:
            dx_world = float(ego_loc.x - self.prev_ego_xy[0])
            dy_world = float(ego_loc.y - self.prev_ego_xy[1])
            ego_fwd = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()
            ego_left_x = -float(ego_right.x)
            ego_left_y = -float(ego_right.y)
            m_long = float(dx_world * float(ego_fwd.x) + dy_world * float(ego_fwd.y))
            m_lat = float(dx_world * ego_left_x + dy_world * ego_left_y)
            if float(math.hypot(m_long, m_lat)) >= float(LANE_DIR_MOTION_MIN_STEP_M):
                motion_dir_xy = (m_long, m_lat)
        self.prev_ego_xy = (float(ego_loc.x), float(ego_loc.y))

        lookahead_dist = float(LANE_LOOKAHEAD_FIXED_M)

        lane, lane_meta = self._estimate_lane_from_points(
            ego_points,
            lane_labels,
            lookahead_dist=lookahead_dist,
            motion_dir_xy=motion_dir_xy,
            prev_lane_dir_sign=self.prev_lane_dir_sign,
            prev_lane_center_offset=self.prev_lane_center_offset,
            prev_lane_width=self.prev_lane_width,
        )
        lane_dir_sign = float(lane_meta.get("lane_dir_sign", 0.0))
        if abs(lane_dir_sign) > 0.5:
            self.prev_lane_dir_sign = 1.0 if lane_dir_sign >= 0.0 else -1.0
        if np.any(np.abs(lane) > 1e-6):
            self.prev_lane_center_offset = float(lane[0])
            if float(lane[2]) > 1e-3:
                self.prev_lane_width = float(lane[2])

        lane_api_assist_used = 0.0
        if USE_API_LANE_ASSIST:
            try:
                lane_prior = compute_lane_gt(carla_env)
                wp = float(np.clip(API_LANE_ASSIST_WEIGHT, 0.0, 1.0))
                ws = float(1.0 - wp)
                lane[0] = float(ws * float(lane[0]) + wp * float(lane_prior[0]))
                lane[1] = float(blend_angle(float(lane[1]), float(lane_prior[1]), wp))
                lane[2] = float(ws * float(lane[2]) + wp * float(lane_prior[2]))
                lane[3] = float(ws * float(lane[3]) + wp * float(lane_prior[3]))
                lane[4] = float(ws * float(lane[4]) + wp * float(lane_prior[4]))
                lane[5] = float(ws * float(lane[5]) + wp * float(lane_prior[5]))
                lane_api_assist_used = 1.0
            except Exception:
                lane_api_assist_used = 0.0

        radar_points = _radar_points_ego(radar)
        ts = float(lidar.timestamp) if lidar is not None else (float(sem_lidar.timestamp) if sem_lidar is not None else 0.0)
        obs, obs_meta = self._estimate_obstacles_from_points(
            ego_points,
            obs_labels,
            ts,
            radar_points=radar_points,
        )
        obs_api_assist_used = 0.0
        if USE_API_OBS_ASSIST:
            try:
                obs_prior = compute_obstacle_gt(
                    carla_env=carla_env,
                    obstacle_k=self.k,
                    obs_range=self.obs_range,
                    front_fwd_min=self.front_fwd_min,
                    front_lat_tol=self.front_lat_tol,
                )
                wp = float(np.clip(API_OBS_ASSIST_WEIGHT, 0.0, 1.0))
                ws = float(1.0 - wp)
                obs = (ws * obs + wp * obs_prior).astype(np.float32)
                obs_api_assist_used = 1.0
            except Exception:
                obs_api_assist_used = 0.0
        lane_sac = canonical_lane_to_sac(lane)
        obs_sac = canonical_obs_to_sac(obs, self.obs_range)
        meta = {
            "lidar_points": float(total_pts),
            "projected_points": float(valid_cnt),
            "projection_valid_ratio": float(valid_ratio),
            "used_lidar_only_fallback": float(used_lidar_only_fallback),
            "lane_nonzero": float(1.0 if np.any(np.abs(lane) > 1e-6) else 0.0),
            "obs_nonzero": float(1.0 if np.any(np.abs(obs) > 1e-6) else 0.0),
            "projection_mode": str(proj_mode),
            "lane_source_sem_lidar": float(1.0 if use_sem_lidar else 0.0),
            "lane_dir_sign": float(lane_meta.get("lane_dir_sign", 0.0)),
            "lane_lookahead_used": float(lane_meta.get("lane_lookahead_used", 0.0)),
            "lane_fit_points": float(lane_meta.get("lane_fit_points", 0.0)),
            "lane_line_slice_count": float(lane_meta.get("lane_line_slice_count", 0.0)),
            "lane_road_fallback_count": float(lane_meta.get("lane_road_fallback_count", 0.0)),
            "lane_used_prev_fallback": float(lane_meta.get("lane_used_prev_fallback", 0.0)),
            "obs_radar_matches": float(obs_meta.get("obs_radar_matches", 0.0)),
            "obs_candidate_clusters": float(obs_meta.get("obs_candidate_clusters", 0.0)),
            "obs_radar_points": float(obs_meta.get("obs_radar_points", 0.0)),
            "lane_api_assist_used": float(lane_api_assist_used),
            "obs_api_assist_used": float(obs_api_assist_used),
        }
        return lane, obs, lane_sac, obs_sac, meta

    def _estimate_lane_from_points(
        self,
        ego_points: np.ndarray,
        labels: np.ndarray,
        lookahead_dist: float,
        motion_dir_xy: Optional[Tuple[float, float]],
        prev_lane_dir_sign: float,
        prev_lane_center_offset: Optional[float],
        prev_lane_width: Optional[float],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        x = ego_points[:, 0]
        y = ego_points[:, 1]
        z = ego_points[:, 2]
        base_mask = (x > 1.0) & (x < 30.0) & (np.abs(y) < 12.0) & (z > -3.0) & (z < 2.0)
        line_slice_count = 0
        road_fallback_count = 0

        line_mask = base_mask & (labels == SEM_ROAD_LINE)
        road_mask = base_mask & np.isin(labels, [SEM_ROAD, SEM_ROAD_LINE])
        if np.count_nonzero(road_mask) < 60:
            # semantic labels may fail in some builds; degrade to geometry-only road hypothesis
            road_mask = base_mask

        xs = []
        centers = []
        widths = []
        for center_x in np.arange(3.0, 24.1, 2.0):
            half = 0.9
            line_sel = line_mask & (x >= center_x - half) & (x <= center_x + half)
            road_sel = road_mask & (x >= center_x - half) & (x <= center_x + half)

            y_left = None
            y_right = None
            width_from_road = False
            if np.count_nonzero(line_sel) >= int(LANE_LINE_SLICE_MIN_POINTS):
                ys_line = y[line_sel]
                left_candidates = ys_line[ys_line > 0.2]
                right_candidates = ys_line[ys_line < -0.2]
                if left_candidates.size >= int(LANE_LINE_SIDE_MIN_POINTS) and right_candidates.size >= int(
                    LANE_LINE_SIDE_MIN_POINTS
                ):
                    y_left = float(np.percentile(left_candidates, 35))
                    y_right = float(np.percentile(right_candidates, 65))

            if (y_left is None or y_right is None) and np.count_nonzero(road_sel) >= 20:
                ys_road = y[road_sel]
                y_left = float(np.percentile(ys_road, 85))
                y_right = float(np.percentile(ys_road, 15))
                width_from_road = True

            if y_left is None or y_right is None:
                continue

            width = y_left - y_right
            if width_from_road:
                if prev_lane_width is not None and np.isfinite(float(prev_lane_width)):
                    lo = max(float(LANE_WIDTH_ROAD_CLAMP_MIN_M), float(prev_lane_width) - 0.7)
                    hi = min(float(LANE_WIDTH_ROAD_CLAMP_MAX_M), float(prev_lane_width) + 0.7)
                    if lo < hi:
                        width = float(np.clip(width, lo, hi))
                width = float(np.clip(width, float(LANE_WIDTH_ROAD_CLAMP_MIN_M), float(LANE_WIDTH_ROAD_CLAMP_MAX_M)))

            if width < float(LANE_WIDTH_VALID_MIN_M) or width > float(LANE_WIDTH_VALID_MAX_M):
                continue

            xs.append(float(center_x))
            centers.append(float(0.5 * (y_left + y_right)))
            widths.append(float(width))
            if width_from_road:
                road_fallback_count += 1
            else:
                line_slice_count += 1

        if len(xs) < 2:
            prev_sign = 1.0 if float(prev_lane_dir_sign) >= 0.0 else -1.0
            lookahead_fb = float(np.clip(lookahead_dist, 3.0, 25.0))
            if prev_lane_center_offset is not None and prev_lane_width is not None:
                lane_center_offset_fb = float(prev_lane_center_offset)
                lane_center_at_ego_fb = -lane_center_offset_fb
                lane_heading_diff_fb = float(0.0 if prev_sign > 0.0 else math.pi)
                lane_fb = np.array(
                    [
                        lane_center_offset_fb,
                        lane_heading_diff_fb,
                        float(prev_lane_width),
                        float(prev_sign * lookahead_fb),
                        float(lane_center_at_ego_fb),
                        0.0,
                    ],
                    dtype=np.float32,
                )
                return lane_fb, {
                    "lane_dir_sign": float(prev_sign),
                    "lane_lookahead_used": float(lookahead_fb),
                    "lane_fit_points": float(len(xs)),
                    "lane_line_slice_count": float(line_slice_count),
                    "lane_road_fallback_count": float(road_fallback_count),
                    "lane_used_prev_fallback": 1.0,
                }
            lane_fallback = np.zeros((6,), dtype=np.float32)
            return lane_fallback, {
                "lane_dir_sign": float(prev_sign),
                "lane_lookahead_used": float(lookahead_fb),
                "lane_fit_points": float(len(xs)),
                "lane_line_slice_count": float(line_slice_count),
                "lane_road_fallback_count": float(road_fallback_count),
                "lane_used_prev_fallback": 0.0,
            }

        x_arr = np.asarray(xs, dtype=np.float32)
        y_arr = np.asarray(centers, dtype=np.float32)
        if len(xs) >= 3:
            poly = np.polyfit(x_arr, y_arr, 2)
            a, b, c = float(poly[0]), float(poly[1]), float(poly[2])
        else:
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            a, b, c = 0.0, float(slope), float(intercept)

        lane_center_at_ego = c
        lane_center_offset = float(-lane_center_at_ego)
        if prev_lane_center_offset is not None and np.isfinite(float(prev_lane_center_offset)):
            lane_center_offset = float(
                float(LANE_CENTER_SMOOTH_ALPHA) * lane_center_offset
                + (1.0 - float(LANE_CENTER_SMOOTH_ALPHA)) * float(prev_lane_center_offset)
            )

        slope0 = float(b)
        dir_vec = np.array([1.0, slope0], dtype=np.float32)
        dir_norm = float(np.linalg.norm(dir_vec))
        if dir_norm < 1e-6:
            dir_vec = np.array([1.0, 0.0], dtype=np.float32)
            dir_norm = 1.0
        dir_vec = dir_vec / dir_norm

        # Sensor-only direction disambiguation:
        # choose lane tangent sign from ego motion, then apply API-GT direction convention.
        dir_sign = 1.0 if float(prev_lane_dir_sign) >= 0.0 else -1.0
        if motion_dir_xy is not None:
            m_long, m_lat = float(motion_dir_xy[0]), float(motion_dir_xy[1])
            dot_pos = float(dir_vec[0] * m_long + dir_vec[1] * m_lat)
            dot_neg = float((-dir_vec[0]) * m_long + (-dir_vec[1]) * m_lat)
            if LANE_DIR_OPPOSITE_MOTION:
                dir_sign = -1.0 if dot_pos >= dot_neg else 1.0
            else:
                dir_sign = 1.0 if dot_pos >= dot_neg else -1.0

        dir_x = float(dir_sign * dir_vec[0])
        dir_y = float(dir_sign * dir_vec[1])
        lane_heading_diff = float(math.atan2(dir_y, dir_x))

        lookahead = float(np.clip(lookahead_dist, 3.0, 25.0))
        x_eval = float(dir_sign * lookahead)
        # Keep lateral evaluation on the observed forward side to avoid large rear-side extrapolation.
        x_eval_lateral = float(lookahead)
        slope_eval = float(2.0 * a * x_eval_lateral + b)

        lane_width = float(np.median(np.asarray(widths, dtype=np.float32))) if widths else 3.5
        if prev_lane_width is not None and np.isfinite(float(prev_lane_width)):
            lane_width = float(
                float(LANE_WIDTH_SMOOTH_ALPHA) * lane_width
                + (1.0 - float(LANE_WIDTH_SMOOTH_ALPHA)) * float(prev_lane_width)
            )
        lane_width = float(np.clip(lane_width, float(LANE_WIDTH_VALID_MIN_M), float(LANE_WIDTH_VALID_MAX_M)))
        next_wp_dx = float(x_eval)
        next_wp_dy = float(a * x_eval_lateral * x_eval_lateral + b * x_eval_lateral + c)
        kappa_base = float((2.0 * a) / max(1e-6, (1.0 + slope_eval * slope_eval) ** 1.5))
        road_curvature = float(dir_sign * kappa_base)

        lane = np.array(
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
        return lane, {
            "lane_dir_sign": float(dir_sign),
            "lane_lookahead_used": float(lookahead),
            "lane_fit_points": float(len(xs)),
            "lane_line_slice_count": float(line_slice_count),
            "lane_road_fallback_count": float(road_fallback_count),
            "lane_used_prev_fallback": 0.0,
        }

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
        radar_points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
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
            return np.zeros((self.k, 3), dtype=np.float32), {
                "obs_radar_matches": 0.0,
                "obs_candidate_clusters": 0.0,
                "obs_radar_points": float(0 if radar_points is None else radar_points.shape[0]),
            }

        clusters_idx = self._cluster_points_grid(candidate[:, :2])
        if not clusters_idx:
            self.prev_clusters = []
            self.prev_ts = ts
            return np.zeros((self.k, 3), dtype=np.float32), {
                "obs_radar_matches": 0.0,
                "obs_candidate_clusters": 0.0,
                "obs_radar_points": float(0 if radar_points is None else radar_points.shape[0]),
            }

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
        radar_match_count = 0
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
            if USE_RADAR_FOR_OBS_SPEED and radar_points is not None and radar_points.shape[0] > 0:
                d2 = (radar_points[:, 0] - cx) * (radar_points[:, 0] - cx) + (radar_points[:, 1] - cy) * (
                    radar_points[:, 1] - cy
                )
                j = int(np.argmin(d2))
                if float(d2[j]) <= float(RADAR_MATCH_RADIUS_M * RADAR_MATCH_RADIUS_M):
                    rel_speed = float(radar_points[j, 2])
                    radar_match_count += 1
            out_entries.append((cx, cy, rel_speed, dist))

        self.prev_clusters = [(e[0], e[1], ts) for e in out_entries]
        self.prev_ts = ts

        selected = sort_obstacles_by_priority(
            entries=[(e[0], e[1], e[2]) for e in out_entries],
            front_fwd_min=self.front_fwd_min,
            front_lat_tol=self.front_lat_tol,
            top_k=self.k,
        )
        return selected, {
            "obs_radar_matches": float(radar_match_count),
            "obs_candidate_clusters": float(len(clusters)),
            "obs_radar_points": float(0 if radar_points is None else radar_points.shape[0]),
        }


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

    # Align next_wp GT semantics with sensor-side fixed lookahead definition.
    # Use map waypoint progression only (no env target waypoint).
    lookahead = float(np.clip(float(LANE_LOOKAHEAD_FIXED_M), 3.0, 25.0))
    next_cands = wp.next(lookahead)
    target_wp = next_cands[0] if next_cands else wp
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


def compute_obstacle_gt_env_obs(
    carla_env: CarlaEnv,
    obstacle_k: int,
    obs_range: float,
    front_fwd_min: float,
    front_lat_tol: float,
) -> np.ndarray:
    """
    GT with the same candidate source priority as current CarlaEnv obstacle obs:
    prefer env._collect_obstacle_candidates() if available.
    """
    ego = carla_env.ego
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_vel = ego.get_velocity()
    ego_fwd = ego_tf.get_forward_vector()
    ego_right = ego_tf.get_right_vector()
    ego_left_x = -float(ego_right.x)
    ego_left_y = -float(ego_right.y)

    candidates = None
    if hasattr(carla_env, "_collect_obstacle_candidates"):
        try:
            candidates = carla_env._collect_obstacle_candidates()
        except Exception:
            candidates = None
    if candidates is None:
        candidates = _collect_world_obstacles(carla_env.world, ego)

    entries: List[Tuple[float, float, float]] = []
    for a in candidates:
        if a is None:
            continue
        if hasattr(a, "id") and int(a.id) == int(ego.id):
            continue
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
        "diag_lane_fit_points",
        "diag_lane_line_slice_count",
        "diag_lane_road_fallback_count",
        "diag_lane_used_prev_fallback",
        "diag_lane_source_sem_lidar",
        "diag_obs_radar_matches",
        "diag_obs_candidate_clusters",
        "diag_obs_radar_points",
        "diag_lane_api_assist_used",
        "diag_obs_api_assist_used",
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
    parser.add_argument(
        "--obstacle-gt-source",
        type=str,
        choices=["world", "env_obs"],
        default="world",
        help="Deprecated. Main obstacle GT is always world API.",
    )

    parser.add_argument("--sem-width", type=int, default=640)
    parser.add_argument("--sem-height", type=int, default=360)
    parser.add_argument("--sem-fov", type=float, default=100.0)
    parser.add_argument("--lidar-range", type=float, default=60.0)
    parser.add_argument("--radar-range", type=float, default=50.0)
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
    obs_errs_world = []
    sac_obs_errs = []
    projection_valid_ratios = []
    lidar_only_fallback_steps = 0
    lane_nonzero_steps = 0
    obs_nonzero_steps = 0
    lane_reverse_dir_steps = 0
    lane_lookahead_used_vals = []
    lane_fit_points_vals = []
    lane_line_slice_vals = []
    lane_road_fallback_vals = []
    lane_used_prev_fallback_steps = 0
    lane_source_sem_lidar_steps = 0
    obs_radar_match_vals = []
    obs_candidate_cluster_vals = []
    obs_radar_points_vals = []
    lane_api_assist_steps = 0
    obs_api_assist_steps = 0

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
    print(f"[MVP] script_version={SCRIPT_VERSION}")
    print("[MVP] main GT standard: lane=map.get_waypoint, obstacle=world.get_actors")
    if str(args.obstacle_gt_source) != "world":
        print("[MVP] warning: --obstacle-gt-source is deprecated; main metric always uses world API GT.")
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
                    radar_range=float(args.radar_range),
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
                    _next_obs, reward, done, _info = env.step(action)
                    total_env_steps += 1
                    env_step_in_ep += 1

                    frame = int(env.world.get_snapshot().frame)
                    sem, lidar, sem_lidar, radar = suite.poll(frame, timeout_s=float(args.sensor_timeout))
                    if sem is None or lidar is None:
                        missing_sensor_frames += 1
                        continue

                    lane_pred, obs_pred_k3, lane_sac_pred, obs_sac_pred_k3, est_meta = estimator.estimate(
                        carla_env=env,
                        sem=sem,
                        lidar=lidar,
                        sem_lidar=sem_lidar,
                        radar=radar,
                    )
                    projection_valid_ratios.append(float(est_meta.get("projection_valid_ratio", 0.0)))
                    lidar_only_fallback_steps += int(est_meta.get("used_lidar_only_fallback", 0.0) > 0.5)
                    lane_nonzero_steps += int(est_meta.get("lane_nonzero", 0.0) > 0.5)
                    obs_nonzero_steps += int(est_meta.get("obs_nonzero", 0.0) > 0.5)
                    lane_reverse_dir_steps += int(est_meta.get("lane_dir_sign", 1.0) < 0.0)
                    lane_lookahead_used_vals.append(float(est_meta.get("lane_lookahead_used", 0.0)))
                    lane_fit_points_vals.append(float(est_meta.get("lane_fit_points", 0.0)))
                    lane_line_slice_vals.append(float(est_meta.get("lane_line_slice_count", 0.0)))
                    lane_road_fallback_vals.append(float(est_meta.get("lane_road_fallback_count", 0.0)))
                    lane_used_prev_fallback_steps += int(est_meta.get("lane_used_prev_fallback", 0.0) > 0.5)
                    lane_source_sem_lidar_steps += int(est_meta.get("lane_source_sem_lidar", 0.0) > 0.5)
                    obs_radar_match_vals.append(float(est_meta.get("obs_radar_matches", 0.0)))
                    obs_candidate_cluster_vals.append(float(est_meta.get("obs_candidate_clusters", 0.0)))
                    obs_radar_points_vals.append(float(est_meta.get("obs_radar_points", 0.0)))
                    lane_api_assist_steps += int(est_meta.get("lane_api_assist_used", 0.0) > 0.5)
                    obs_api_assist_steps += int(est_meta.get("obs_api_assist_used", 0.0) > 0.5)

                    lane_gt = compute_lane_gt(env)
                    obs_gt_world_k3 = compute_obstacle_gt(
                        carla_env=env,
                        obstacle_k=int(args.obstacle_k),
                        obs_range=float(args.obs_range),
                        front_fwd_min=float(args.front_fwd_min),
                        front_lat_tol=float(args.front_lat_tol),
                    )
                    obs_gt_k3 = obs_gt_world_k3

                    # SAC-style metrics computed from API GT (not from CarlaEnv observation slices).
                    lane_sac_gt = canonical_lane_to_sac(lane_gt)
                    obs_sac_gt_k3 = canonical_obs_to_sac(obs_gt_k3, float(args.obs_range))

                    lane_abs = np.abs(lane_pred - lane_gt)
                    lane_abs[1] = abs(wrap_angle(float(lane_pred[1]) - float(lane_gt[1])))
                    obs_abs = np.abs(obs_pred_k3 - obs_gt_k3)
                    obs_abs_world = np.abs(obs_pred_k3 - obs_gt_world_k3)
                    lane_sac_abs = np.abs(lane_sac_pred - lane_sac_gt)
                    obs_sac_abs = np.abs(obs_sac_pred_k3 - obs_sac_gt_k3)

                    lane_errs.append(lane_abs)
                    obs_errs.append(obs_abs)
                    obs_errs_world.append(obs_abs_world)
                    sac_lane_errs.append(lane_sac_abs)
                    sac_obs_errs.append(obs_sac_abs)

                    row = {
                        "episode": int(episode_idx),
                        "env_step": int(env_step_in_ep),
                        "frame": int(frame),
                        "reward": float(reward),
                        "done": int(done),
                        "diag_lane_fit_points": float(est_meta.get("lane_fit_points", 0.0)),
                        "diag_lane_line_slice_count": float(est_meta.get("lane_line_slice_count", 0.0)),
                        "diag_lane_road_fallback_count": float(est_meta.get("lane_road_fallback_count", 0.0)),
                        "diag_lane_used_prev_fallback": float(est_meta.get("lane_used_prev_fallback", 0.0)),
                        "diag_lane_source_sem_lidar": float(est_meta.get("lane_source_sem_lidar", 0.0)),
                        "diag_obs_radar_matches": float(est_meta.get("obs_radar_matches", 0.0)),
                        "diag_obs_candidate_clusters": float(est_meta.get("obs_candidate_clusters", 0.0)),
                        "diag_obs_radar_points": float(est_meta.get("obs_radar_points", 0.0)),
                        "diag_lane_api_assist_used": float(est_meta.get("lane_api_assist_used", 0.0)),
                        "diag_obs_api_assist_used": float(est_meta.get("obs_api_assist_used", 0.0)),
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
    obs_mae_world = np.mean(np.stack(obs_errs_world, axis=0), axis=(0, 1))
    sac_lane_mae = np.mean(np.stack(sac_lane_errs, axis=0), axis=0)
    sac_obs_mae = np.mean(np.stack(sac_obs_errs, axis=0), axis=(0, 1))
    proj_ratio_mean = float(np.mean(projection_valid_ratios)) if projection_valid_ratios else 0.0
    lookahead_mean = float(np.mean(lane_lookahead_used_vals)) if lane_lookahead_used_vals else 0.0
    lane_fit_points_mean = float(np.mean(lane_fit_points_vals)) if lane_fit_points_vals else 0.0
    lane_line_slice_mean = float(np.mean(lane_line_slice_vals)) if lane_line_slice_vals else 0.0
    lane_road_fallback_mean = float(np.mean(lane_road_fallback_vals)) if lane_road_fallback_vals else 0.0
    obs_radar_matches_mean = float(np.mean(obs_radar_match_vals)) if obs_radar_match_vals else 0.0
    obs_candidate_clusters_mean = float(np.mean(obs_candidate_cluster_vals)) if obs_candidate_cluster_vals else 0.0
    obs_radar_points_mean = float(np.mean(obs_radar_points_vals)) if obs_radar_points_vals else 0.0

    summary = {
        "scenario": str(args.scenario),
        "map": str(args.map),
        "script_version": SCRIPT_VERSION,
        "gt_main_source": {
            "lane": "carla.map.get_waypoint (+ wp.next fixed lookahead for next_wp)",
            "obstacle": "carla.world.get_actors",
        },
        "obstacle_gt_source": "world",
        "valid_steps": int(valid_steps),
        "total_env_steps": int(total_env_steps),
        "missing_sensor_frames": int(missing_sensor_frames),
        "lane_mae": {k: float(v) for k, v in zip(LANE_NAMES, lane_mae.tolist())},
        "obstacle_mae": {k: float(v) for k, v in zip(OBS_NAMES, obs_mae.tolist())},
        "obstacle_mae_world": {k: float(v) for k, v in zip(OBS_NAMES, obs_mae_world.tolist())},
        "sac_lane_mae": {k: float(v) for k, v in zip(SAC_LANE_NAMES, sac_lane_mae.tolist())},
        "sac_obstacle_mae": {k: float(v) for k, v in zip(SAC_OBS_NAMES, sac_obs_mae.tolist())},
        "projection_valid_ratio_mean": float(proj_ratio_mean),
        "lidar_only_fallback_steps": int(lidar_only_fallback_steps),
        "lane_nonzero_rate": float(lane_nonzero_steps / max(1, valid_steps)),
        "lane_reverse_dir_steps": int(lane_reverse_dir_steps),
        "lane_reverse_dir_rate": float(lane_reverse_dir_steps / max(1, valid_steps)),
        "lane_lookahead_used_mean": float(lookahead_mean),
        "lane_fit_points_mean": float(lane_fit_points_mean),
        "lane_line_slice_mean": float(lane_line_slice_mean),
        "lane_road_fallback_mean": float(lane_road_fallback_mean),
        "lane_used_prev_fallback_rate": float(lane_used_prev_fallback_steps / max(1, valid_steps)),
        "lane_source_sem_lidar_rate": float(lane_source_sem_lidar_steps / max(1, valid_steps)),
        "obs_radar_matches_mean": float(obs_radar_matches_mean),
        "obs_candidate_clusters_mean": float(obs_candidate_clusters_mean),
        "obs_radar_points_mean": float(obs_radar_points_mean),
        "lane_api_assist_rate": float(lane_api_assist_steps / max(1, valid_steps)),
        "obs_api_assist_rate": float(obs_api_assist_steps / max(1, valid_steps)),
        "api_assist_weights": {
            "lane": float(API_LANE_ASSIST_WEIGHT),
            "obstacle": float(API_OBS_ASSIST_WEIGHT),
        },
        "obstacle_nonzero_rate": float(obs_nonzero_steps / max(1, valid_steps)),
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
