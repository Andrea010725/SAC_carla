"""Proximal Policy Optimization Agent"""

import os
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random

from typing import Union, Dict, Any, Optional, Tuple

from .. import utils
from ..agents.agents import Agent
from ..parameters import DynamicParameter
from ..networks.networks import PPONetwork

from tensorflow.keras import losses
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import ipdb
import json
from dataclasses import dataclass
import logging

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:  # Avoid duplicate handlers
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class TrainingConfig:
    """Configuration class for PPO training"""
    episodes: int
    timesteps: int
    save_every: Union[bool, str, int] = False
    render_every: Union[bool, str, int] = False
    close: bool = True


@dataclass
class ActionData:
    """Container for action-related data from policy."""
    action: Any
    action_env: Any
    mean: Any
    std: Any
    value: Any
    log_prob: Any


@dataclass
class TransitionData:
    """Container for environment transition data."""
    next_state: Any
    reward: float
    done: bool
    action_data: ActionData


@dataclass
class EpisodeStats:
    """Container for episode statistics."""
    start_time: float
    total_reward: float = 0.0
    steps: int = 0

    def update(self, reward: float) -> None:
        """Update episode statistics with new reward."""
        self.total_reward += reward
        self.steps += 1

    @property
    def duration(self) -> float:
        """Calculate episode duration in seconds."""
        return time.time() - self.start_time


class PPOAgent(Agent):
    # TODO: dynamic-parameters: gamma, lambda, opt_steps, update_freq?, polyak?, clip_norm
    # TODO: debug each action separately
    # TODO: RNN support
    def __init__(self, *args, policy_lr: Union[float, LearningRateSchedule, DynamicParameter] = 1e-3, gamma=0.99,
                 lambda_=0.95, value_lr: Union[float, LearningRateSchedule, DynamicParameter] = 3e-4, load=False,
                 optimization_steps=(3, 3), name='ppo-agent', optimizer='adam', clip_norm=(1.0, 1.0),
                 clip_ratio: Union[float, LearningRateSchedule, DynamicParameter] = 0.2, seed_regularization=False,
                 entropy_regularization: Union[float, LearningRateSchedule, DynamicParameter] = 0.0,
                 network: Union[dict, PPONetwork] = None, update_frequency=1, polyak=1.0, repeat_action=1,
                 advantage_scale: Union[float, LearningRateSchedule, DynamicParameter] = 2.0, **kwargs):
        assert 0.0 < polyak <= 1.0
        assert repeat_action >= 1

        kwargs.pop("town", None)
        super().__init__(*args, name=name, **kwargs)

        self.memory: PPOMemory = None
        self.gamma = gamma
        self.lambda_ = lambda_
        self.repeat_action = repeat_action
        self.adv_scale = DynamicParameter.create(value=advantage_scale)

        # ✅ 控制是否打印 policy debug（默认关闭，避免刷屏）
        self.debug_policy_print = bool(kwargs.get("debug_policy_print", False))

        # TRD (Temporal Return Decomposition) loss coefficient
        self.trd_loss_coef = kwargs.get('trd_loss_coef', 0.01)

        # TRD schedule
        self.trd_coef_max = float(kwargs.get("trd_loss_coef", 0.01))
        self.trd_warmup_updates = int(kwargs.get("trd_warmup_updates", 20))  # 前20次update不训练TRD
        self.trd_ramp_updates = int(kwargs.get("trd_ramp_updates", 80))  # 再用80次线性爬升到max
        self.update_step = tf.Variable(0, dtype=tf.int64, trainable=False)

        if seed_regularization:
            def _seed_regularization():
                seed = random.randint(a=0, b=2 ** 32 - 1)
                self.set_random_seed(seed)

            self.seed_regularization = _seed_regularization
            self.seed_regularization()
        else:
            self.seed_regularization = lambda: None

        # Entropy regularization
        # self.entropy_strength = DynamicParameter.create(value=float(entropy_regularization))
        self.entropy_strength = DynamicParameter.create(value=float(entropy_regularization))

        # Ratio clipping
        if isinstance(clip_ratio, float):
            assert clip_ratio >= 0.0

        self.clip_ratio = DynamicParameter.create(value=clip_ratio)

        # Action space
        self._init_action_space()

        print('state_spec:', self.state_spec)
        print('action_shape:', self.num_actions)
        print('distribution:', self.distribution_type)

        # Gradient clipping:
        self._init_gradient_clipping(clip_norm)

        # Networks & Loading
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                                 value=os.path.join(self.base_path, 'value_net'))

        if isinstance(network, dict):
            network_class = network.pop('network', PPONetwork)

            if network_class is PPONetwork:
                # policy/value-specific arguments
                policy_args = network.pop('policy', {})
                value_args = network.pop('value', policy_args)

                # common arguments
                for k, v in network.items():
                    if k not in policy_args:
                        policy_args[k] = v
                    if k not in value_args:
                        value_args[k] = v

                self.network = network_class(agent=self, policy=policy_args, value=value_args, **network)
            else:
                self.network = network_class(agent=self, **network)
        else:
            self.network = PPONetwork(agent=self, policy={}, value={})

        # Optimization
        self.update_frequency = update_frequency
        self.policy_lr = DynamicParameter.create(value=policy_lr)
        self.value_lr = DynamicParameter.create(value=value_lr)
        self.optimization_steps = dict(policy=optimization_steps[0], value=optimization_steps[1])

        self.policy_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.policy_lr)
        self.value_optimizer = utils.get_optimizer_by_name(optimizer, learning_rate=self.value_lr)

        self.should_polyak_average = polyak < 1.0
        self.polyak_coeff = polyak

        if load:
            self.load()

    # =========================
    # ✅ NEW: 两段 policy 的统一变量列表（lat + lon）
    # =========================
    def _policy_vars(self):
        return (self.network.policy_lat.trainable_variables +
                self.network.policy_lon.trainable_variables)

    def _init_gradient_clipping(self, clip_norm: Union[tuple, float, None]):
        if clip_norm is None:
            self.should_clip_policy_grads = False
            self.should_clip_value_grads = False

        elif isinstance(clip_norm, float):
            assert clip_norm > 0.0
            self.should_clip_policy_grads = True
            self.should_clip_value_grads = True

            self.grad_norm_policy = clip_norm
            self.grad_norm_value = clip_norm
        else:
            assert isinstance(clip_norm, tuple)

            if clip_norm[0] is None:
                self.should_clip_policy_grads = False
            else:
                assert isinstance(clip_norm[0], float)
                assert clip_norm[0] > 0.0
                self.should_clip_policy_grads = True
                self.grad_norm_policy = tf.constant(clip_norm[0], dtype=tf.float32)

            if clip_norm[1] is None:
                self.should_clip_value_grads = False
            else:
                assert isinstance(clip_norm[1], float)
                assert clip_norm[1] > 0.0
                self.should_clip_value_grads = True
                self.grad_norm_value = tf.constant(clip_norm[1], dtype=tf.float32)

    # TODO: handle complex action spaces (make use of Agent.action_spec)
    def _init_action_space(self):
        action_space = self.env.action_space

        if isinstance(action_space, gym.spaces.Box):
            # continuous
            self.num_actions = int(action_space.shape[0])

            # ✅ 你这里已经强制用 gaussian（Normal），不要再走 beta 缩放
            self.distribution_type = 'gaussian'

            # bounds (for safe clipping before sending to env)
            self.action_low = tf.constant(action_space.low, dtype=tf.float32)
            self.action_high = tf.constant(action_space.high, dtype=tf.float32)

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

            self.convert_action = _convert_action

        else:
            # discrete
            self.distribution_type = 'categorical'

            if isinstance(action_space, gym.spaces.MultiDiscrete):
                assert np.all(action_space.nvec == action_space.nvec[0])
                self.num_actions = int(action_space.nvec.shape[0])
                self.num_classes = int(action_space.nvec[0] + 1)

                def _convert_action(a):
                    a = tf.convert_to_tensor(a)
                    # a could be shape (1, A) or (A,)
                    if a.shape.rank == 2:
                        a = a[0]
                    return tf.cast(a, dtype=tf.int32).numpy()

                self.convert_action = _convert_action
            else:
                self.num_actions = 1
                self.num_classes = int(action_space.n)

                def _convert_action(a):
                    a = tf.convert_to_tensor(a)
                    return tf.cast(tf.squeeze(a), dtype=tf.int32).numpy()

                self.convert_action = _convert_action

    # =========================
    # ✅ FIX: act 不能再调用 network.act（里面引用 self.policy 会炸）
    # =========================
    def act(self, state, *args, **kwargs):
        action, mean, std, log_prob, value = self.network.predict(inputs=state)
        return self.convert_action(action)

    def predict(self, state, *args, **kwargs):
        return self.network.predict(inputs=state)

    def update(self):
        t0 = time.time()
        self.seed_regularization()

        # Prepare data:
        value_batches = self.get_value_batches()
        policy_batches = self.get_policy_batches()

        # Policy network optimization:
        for opt_step in range(self.optimization_steps['policy']):
            for data_batch in policy_batches:
                self.seed_regularization()
                total_loss, policy_grads = self.get_policy_gradients(data_batch)
                self.update_policy(policy_grads)

                if isinstance(policy_grads, dict):
                    policy_grads = policy_grads['policy']

                # 注意：policy_grads 里可能有 None
                global_norm = tf.linalg.global_norm([g for g in policy_grads if g is not None])
                self.log(grad_norm_policy=global_norm)

        # Value network optimization:
        for _ in range(self.optimization_steps['value']):
            for data_batch in value_batches:
                self.seed_regularization()
                value_loss, value_grads = self.get_value_gradients(data_batch)

                self.update_value(value_grads)

                if isinstance(value_grads, dict):
                    value_grads = value_grads['value']

                grads_norm = []
                for g in value_grads:
                    if g is None:
                        grads_norm.append(tf.constant(0.0, dtype=tf.float32))
                    else:
                        grads_norm.append(tf.norm(g))

                self.log(loss_value=value_loss, lr_value=self.value_lr.value,
                         gradients_norm_value=grads_norm)

        print(f'Update took {round(time.time() - t0, 3)}s')

        # ==== 新增: 偶尔 dump 一次 TRD 验证 ====
        try:
            self.dump_trd_example(filename="trd_debug.txt")
        except Exception as e:
            print(f"dump_trd_example failed: {e}")

        self.update_step.assign_add(1)

    # =========================
    # ✅ FIX: policy grads 必须对 (policy_lat + policy_lon) 求梯度
    # 且 policy_objective 返回 (loss, kl)，这里要取 loss
    # =========================
    def get_policy_gradients(self, batch):
        with tf.GradientTape() as tape:
            out = self.policy_objective(batch)
            if isinstance(out, (tuple, list)):
                loss = out[0]
            else:
                loss = out

        vars_ = self._policy_vars()
        gradients = tape.gradient(loss, vars_)
        return loss, gradients

    def update_policy(self, gradients) -> (list, bool):
        return self.apply_policy_gradients(gradients), True

    # =========================
    # ✅ FIX: apply_gradients 不能再用 self.network.policy
    # 同时支持 polyak，对 lat/lon 两套分别 polyak
    # =========================
    def apply_policy_gradients(self, gradients):
        vars_ = self._policy_vars()

        if self.should_clip_policy_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_policy)

        # 过滤 None grads
        gv = [(g, v) for g, v in zip(gradients, vars_) if g is not None]
        if len(gv) == 0:
            print("[PPO] ⚠️ all policy grads are None, skip apply_gradients")
            # 仍然保持 old_policy 同步（避免下一步 act/predict 用旧到离谱的权重）
            self.network.update_old_policy()
            return gradients

        if self.should_polyak_average:
            old_lat = self.network.policy_lat.get_weights()
            old_lon = self.network.policy_lon.get_weights()

            self.policy_optimizer.apply_gradients(gv)

            # 对两套 policy 分别做 polyak
            utils.polyak_averaging(self.network.policy_lat, old_lat, alpha=self.polyak_coeff)
            utils.polyak_averaging(self.network.policy_lon, old_lon, alpha=self.polyak_coeff)

            # 同步 old_policy_lat/lon
            self.network.update_old_policy()
        else:
            self.policy_optimizer.apply_gradients(gv)
            self.network.update_old_policy()

        return gradients

    def get_value_gradients(self, batch):
        with tf.GradientTape() as tape:
            loss = self.value_objective(batch)
        gradients = tape.gradient(loss, self.network.value.trainable_variables)
        return loss, gradients

    def update_value(self, gradients) -> (list, bool):
        return self.apply_value_gradients(gradients), True

    def apply_value_gradients(self, gradients):
        if self.should_clip_value_grads:
            gradients = utils.clip_gradients(gradients, norm=self.grad_norm_value)

        # 过滤 None grads
        gv = [(g, v) for g, v in zip(gradients, self.network.value.trainable_variables) if g is not None]
        if len(gv) == 0:
            print("[PPO] ⚠️ all value grads are None, skip apply_gradients")
            return gradients

        if self.should_polyak_average:
            old_weights = self.network.value.get_weights()
            self.value_optimizer.apply_gradients(gv)
            utils.polyak_averaging(self.network.value, old_weights, alpha=self.polyak_coeff)
        else:
            self.value_optimizer.apply_gradients(gv)

        return gradients

    def value_batch_tensors(self):
        trd = self.memory.trd_targets
        if trd is None:
            # 用 0 占位，shape 需要和 value 网络输出的 trd_pred 对齐（比如 [T, B]）
            T = tf.shape(self.memory.returns)[0]
            B = getattr(self.network, "trd_bins", 10) + 1
            trd = tf.zeros([T, B], dtype=tf.float32)
        return self.memory.states, self.memory.returns, trd

    def policy_batch_tensors(self) -> Union[tuple, dict]:
        """Defines which data to use in `get_policy_batches()`"""
        return self.memory.states, self.memory.advantages, self.memory.actions, self.memory.log_probabilities

    def get_value_batches(self):
        """Computes batches of data for updating the value network"""
        return utils.data_to_batches(tensors=self.value_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_remainder, skip=self.skip_count,
                                     shuffle=True, shuffle_batches=False, num_shards=self.obs_skipping)

    def get_policy_batches(self):
        """Computes batches of data for updating the policy network"""
        return utils.data_to_batches(tensors=self.policy_batch_tensors(), batch_size=self.batch_size,
                                     drop_remainder=self.drop_batch_remainder, skip=self.skip_count,
                                     num_shards=self.obs_skipping, shuffle=self.shuffle,
                                     shuffle_batches=self.shuffle_batches)

    # TRD
    @tf.function
    @tf.function
    def value_objective(self, batch):
        states, returns, trd_targets = batch[:3]
        values, trd_pred = self.network.value_and_trd(states, training=True)

        base_loss = tf.reduce_mean(losses.MSE(y_true=returns[:, 0], y_pred=values[:, 0]))
        exp_loss = tf.reduce_mean(losses.MSE(y_true=returns[:, 1], y_pred=values[:, 1]))
        value_loss = 0.5 * (0.25 * base_loss + exp_loss / (self.network.exp_scale ** 2))

        # ---- TRD coef schedule (warmup + ramp) ----
        step = tf.cast(self.update_step, tf.float32)
        warm = tf.cast(self.trd_warmup_updates, tf.float32)
        ramp = tf.cast(self.trd_ramp_updates, tf.float32)
        maxc = tf.constant(self.trd_coef_max, tf.float32)

        # warmup: coef=0
        # ramp: coef = maxc * clip((step-warm)/ramp, 0..1)
        prog = tf.where(ramp > 0.0, (step - warm) / ramp, 1.0)
        prog = tf.clip_by_value(prog, 0.0, 1.0)
        coef = maxc * prog

        # ---- TRD loss ----
        if (trd_targets is None) or (not tf.is_tensor(trd_targets)):
            trd_loss = tf.constant(0.0, tf.float32)
            total_loss = value_loss
        else:
            b = tf.minimum(tf.shape(trd_pred)[0], tf.shape(trd_targets)[0])
            trd_loss = tf.reduce_mean(tf.square(trd_pred[:b] - trd_targets[:b]))
            total_loss = value_loss + coef * trd_loss

        self.log(loss_value_base=base_loss,
                 loss_value_exp=exp_loss,
                 loss_trd=trd_loss,
                 trd_loss_coef=coef)

        return total_loss

    def policy_objective(self, batch):
        """PPO-Clip Objective (Leader-Follower with y_ref message)"""
        states, advantages, actions, old_log_probabilities = batch[:4]

        # new_log_prob_joint: [B], entropy: scalar
        new_logp, entropy = self.network.log_prob_and_entropy(states, actions, training=True)

        # --------- 1) joint logprob 统一成 [B] ---------
        def _to_vec(x):
            x = tf.convert_to_tensor(x)
            if x.shape.rank == 2:  # [B, K]
                x = tf.reduce_sum(x, axis=1)
            return tf.reshape(x, [-1])  # [B]

        old_logp = _to_vec(old_log_probabilities)
        new_logp = _to_vec(new_logp)

        old_logp = tf.where(tf.math.is_finite(old_logp), old_logp, tf.zeros_like(old_logp))
        new_logp = tf.where(tf.math.is_finite(new_logp), new_logp, tf.zeros_like(new_logp))

        # --------- 2) advantages 统一成 [B] + stop_gradient ---------
        adv = tf.convert_to_tensor(advantages, dtype=tf.float32)
        adv = tf.reshape(adv, [-1])  # [B]
        adv = tf.stop_gradient(adv)

        # （推荐）adv 标准化，能明显稳定 PPO
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

        adv = tf.where(tf.math.is_finite(adv), adv, tf.zeros_like(adv))

        # --------- 3) ratio 数值稳定：clip log_ratio 防 exp 溢出 ---------
        log_ratio = new_logp - old_logp
        log_ratio = tf.clip_by_value(log_ratio, -20.0, 20.0)  # 关键：防 Inf/NaN
        ratio = tf.exp(log_ratio)

        clip_value = self.clip_ratio()
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_value, 1.0 + clip_value)

        # policy loss
        pg1 = ratio * adv
        pg2 = clipped_ratio * adv
        policy_loss = -tf.reduce_mean(tf.minimum(pg1, pg2))

        # --------- 4) approx KL & clipfrac（诊断 PPO 是否有效）---------
        approx_kl = tf.reduce_mean(tf.stop_gradient(old_logp - new_logp))
        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), clip_value), tf.float32))

        # --------- 5) entropy bonus（注意：entropy 是 scalar）---------
        # （强烈建议）check 数值，别让 logger 把错误吞掉
        tf.debugging.check_numerics(policy_loss, "policy_loss has NaN/Inf")
        tf.debugging.check_numerics(entropy, "entropy has NaN/Inf")
        tf.debugging.check_numerics(approx_kl, "approx_kl has NaN/Inf")

        # ✅ 仅在需要时打印（默认关闭，避免刷屏/拖慢训练）
        if getattr(self, "debug_policy_print", False):
            tf.print("[DBG] states", tf.shape(states), "actions", tf.shape(actions), "old_logp",
                     tf.shape(old_log_probabilities))

        entropy_bonus = self.entropy_strength() * entropy
        total_loss = policy_loss - entropy_bonus

        # --------- 6) logging：不要全吞；至少把异常打印出来 ---------
        try:
            self.log(
                ratio=tf.reduce_mean(ratio),
                log_prob=tf.reduce_mean(new_logp),
                entropy=entropy,
                entropy_coeff=(self.entropy_strength.value
                               if hasattr(self.entropy_strength, "value")
                               else self.entropy_strength()),
                ratio_clip=clip_value,
                kl_divergence=approx_kl,
                clipfrac=clipfrac,
                log_ratio_mean=tf.reduce_mean(log_ratio),
                log_ratio_var=tf.math.reduce_variance(log_ratio),
                loss_policy=policy_loss,
                loss_entropy=entropy_bonus,
            )
        except Exception as e:
            # 关键：别 silent fail
            tf.print("[policy_objective][log error]:", e)

        return total_loss, approx_kl

    # =========================
    # ✅ FIX: collect() 里原来调用 network.act2 会炸（PPONetwork 没有 self.policy）
    # 这里改成统一用 network.predict
    # =========================
    def collect(self, episodes: int, timesteps: int, render=True, record_threshold=0.0, seeds=None, close=True):
        import random
        sample_seed = False

        if isinstance(seeds, int):
            self.set_random_seed(seed=seeds)
        elif isinstance(seeds, list):
            sample_seed = True

        for episode in range(1, episodes + 1):
            if sample_seed:
                self.set_random_seed(seed=random.choice(seeds))

            self.reset()
            episode_reward = 0.0
            memory = PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)

            state = self.env.reset()
            state = utils.to_tensor(state)

            if isinstance(state, dict):
                state = {f'state_{k}': v for k, v in state.items()}

            for t in range(1, timesteps + 1):
                if render:
                    self.env.render()

                action, mean, std, log_prob, value = self.network.predict(state)
                next_state, reward, done, _ = self.env.step(self.convert_action(action))
                episode_reward += reward

                self.log(actions=action, rewards=reward, values=value, log_probs=log_prob)

                memory.append(state, action, reward, value, log_prob)
                state = utils.to_tensor(next_state)

                if isinstance(state, dict):
                    state = {f'state_{k}': v for k, v in state.items()}

                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps with reward {episode_reward}.')
                    last_value = self.network.predict_last_value(state, timestep=(t + 1) / timesteps, is_terminal=done)
                    memory.end_trajectory(last_value)
                    break

            self.log(evaluation_reward=episode_reward)
            self.write_summaries()

            if episode_reward >= record_threshold:
                memory.serialize(episode, save_path=self.traces_dir)

        if close:
            self.env.close()

    def learn(self, episodes: int, timesteps: int, save_every: Union[bool, str, int] = False,
              render_every: Union[bool, str, int] = False, close=True):
        assert episodes % self.update_frequency == 0

        if (save_every is False) or (save_every is None):
            save_every = episodes + 1
        elif save_every is True:
            save_every = 1
        elif save_every == 'end':
            save_every = episodes
        else:
            assert episodes % save_every == 0

        if render_every is False:
            render_every = episodes + 1
        elif render_every is True:
            render_every = 1

        try:
            self.memory = self.get_memory()
            for episode in range(1, episodes + 1):
                self.seed_regularization()
                self.on_episode_start()

                preprocess_fn = self.preprocess()
                self.reset()

                state = self.env.reset()
                episode_reward = 0.0
                t0 = time.time()
                render = episode % render_every == 0

                for t in range(1, timesteps + 1):
                    if render:
                        self.env.render()

                    if isinstance(state, dict):
                        state = {f'state_{k}': v for k, v in state.items()}

                    state = preprocess_fn(state)
                    state = utils.to_tensor(state)

                    # Agent prediction
                    action, mean, std, log_prob, value = self.predict(state)
                    action_env = self.convert_action(action)

                    # debug
                    if t % 50 == 1:
                        try:
                            print(f"�� [STEP {t}] action网络输出: {action.numpy()}")
                            print(f"�� [STEP {t}] action_env转换后: {action_env}")
                            print(f"�� [STEP {t}] mean: {mean.numpy()}, std: {std.numpy()}")
                        except Exception:
                            pass

                    # Environment step
                    for _ in range(self.repeat_action):
                        next_state, reward, done, _ = self.env.step(action_env)
                        episode_reward += reward
                        if done:
                            break

                    self.log(actions=action, action_env=action_env, rewards=reward,
                             distribution_mean=mean, distribution_std=std)

                    # self.memory.append(state, action, reward, value, log_prob)
                    self.memory.append(state, action, reward, value, log_prob)

                    state = next_state


                    if done or (t == timesteps):
                        print(f'Episode {episode} terminated after {t} timesteps in {round((time.time() - t0), 3)}s '
                              f'with reward {round(episode_reward, 3)}.')
                        self.log(timestep=t)

                        if isinstance(state, dict):
                            state = {f'state_{k}': v for k, v in state.items()}

                        state = preprocess_fn(state)
                        state = utils.to_tensor(state)

                        last_value = self.network.predict_last_value(state, timestep=(t + 1) / timesteps,
                                                                     is_terminal=done)
                        self.end_episode(last_value, append=self.update_frequency > 1)
                        break

                if episode % self.update_frequency == 0:
                    self.update()
                    self.memory.delete()
                    self.memory = self.get_memory()

                elif self.update_frequency > 1:
                    self.memory.rewards = self.memory.rewards[:-1]
                    self.memory.values = self.memory.values[:-1]
                    # ✅ 重要：去掉 dummy 后同步 index，避免切片错位
                    if hasattr(self.memory, "index"):
                        self.memory.index = max(0, self.memory.index - 1)

                self.log(episode_rewards=episode_reward)
                self.write_summaries()

                if self.should_record:
                    self.record(episode)

                self.on_episode_end()

                if episode % save_every == 0:
                    self.save()
        finally:
            if close:
                print('closing...')
                self.env.close()

    def get_memory(self):
        """Instantiate the agent's memory; easy to subclass"""
        return PPOMemory(state_spec=self.state_spec, num_actions=self.num_actions)

    # TRD
    def end_episode(self, last_value, append=False):
        """Used during learning `learn(...)` to terminate an episode"""
        self.memory.end_trajectory(last_value)

        # 1) returns
        returns_scalar, returns_decomp = self.memory.compute_returns(discount=self.gamma, append=append)

        # 2) TRD targets
        if hasattr(self.memory, "compute_trd_targets"):
            try:
                n_bins = getattr(self.network, "trd_bins", 10)
                self.memory.compute_trd_targets(discount=self.gamma, append=append, n_bins=n_bins)
            except Exception as e:
                print(f"[TRD] compute_trd_targets failed: {e}")

        # 3) advantages
        values, advantages = self.memory.compute_advantages(
            self.gamma, self.lambda_, scale=self.adv_scale(), append=append
        )

        # =========================
        # ✅ 绝对鲁棒对齐（加入 states/logp 的长度）
        # =========================

        # 1) states length
        if self.memory.simple_state:
            T_states = tf.shape(self.memory.states)[0]
        else:
            # dict state: 取最小长度
            T_states = tf.reduce_min(
                tf.stack([tf.shape(v)[0] for v in self.memory.states.values()])
            )

        # 2) actions/logp length
        T_actions = tf.shape(self.memory.actions)[0]
        T_logp    = tf.shape(self.memory.log_probabilities)[0]

        # 3) returns（当前 episode 的标量回报）
        returns_scalar = tf.convert_to_tensor(returns_scalar, dtype=tf.float32)

        # 4) values / adv（当前 episode）
        values_full = tf.convert_to_tensor(values, dtype=tf.float32)
        values_t = values_full[:-1]  # 去掉 bootstrap

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        # ✅ 对齐长度：当 append=True（多 episode 累积）时，
        # 需要用“累计长度”来对齐，否则会把旧 episode 的数据截断掉，
        # 导致 states/actions/returns 长度不一致。
        if append:
            Tr = tf.shape(self.memory.returns)[0] if self.memory.returns is not None else tf.shape(returns_scalar)[0]
            Ta = tf.shape(self.memory.advantages)[0] if self.memory.advantages is not None else tf.shape(advantages)[0]
            Tv_t = tf.maximum(tf.shape(self.memory.values)[0] - 1, 0) if self.memory.values is not None else tf.shape(values_t)[0]
        else:
            Tr = tf.shape(returns_scalar)[0]
            Ta = tf.shape(advantages)[0]
            Tv_t = tf.shape(values_t)[0]

        # ✅ L = 所有关键量的最小长度（把 states/logp 也算进去）
        L = tf.reduce_min(tf.stack([T_states, T_actions, T_logp, Tr, Tv_t, Ta]))

        # ---- 统一裁剪到 L ----
        if self.memory.simple_state:
            self.memory.states = self.memory.states[:L]
        else:
            self.memory.states = {k: v[:L] for k, v in self.memory.states.items()}

        self.memory.actions = self.memory.actions[:L]
        self.memory.log_probabilities = self.memory.log_probabilities[:L]

        values_t = values_t[:L]
        advantages = advantages[:L]
        returns_scalar = returns_scalar[:L]

        # 这些如果存在也裁
        if self.memory.returns is not None:
            self.memory.returns = self.memory.returns[:L]
        if self.memory.advantages is not None:
            self.memory.advantages = self.memory.advantages[:L]
        if getattr(self.memory, "trd_targets", None) is not None:
            self.memory.trd_targets = self.memory.trd_targets[:L]

        # rewards/values 一般是 L+1（含 bootstrap/dummy），裁成 L+1 更安全
        if tf.shape(self.memory.rewards)[0] > L + 1:
            self.memory.rewards = self.memory.rewards[:L + 1]
        if tf.shape(self.memory.values)[0] > L + 1:
            self.memory.values = self.memory.values[:L + 1]

        print("[ALIGN]",
              "L=", int(L.numpy()),
              "states=", int(T_states.numpy()),
              "actions=", int(T_actions.numpy()),
              "logp=", int(T_logp.numpy()),
              "returns=", int(Tr.numpy()),
              "values_t=", int(Tv_t.numpy()),
              "adv=", int(Ta.numpy()))


        # ✅ returns - values_t 绝不会再 shape mismatch
        self.log(
            returns=returns_scalar,
            advantages=advantages,
            values=values_t,
            returns_minus_values=returns_scalar- values_t,
            returns_base=self.memory.returns[:L, 0] if self.memory.returns is not None else None,
            returns_exp=self.memory.returns[:L, 1] if self.memory.returns is not None else None,
            values_base=self.memory.values[:L, 0] if tf.shape(self.memory.values)[0] >= L + 1 else None,
            values_exp = self.memory.values[:L, 1] if tf.shape(self.memory.values)[0] >= L + 1 else None,
            advantages_normalized=self.memory.advantages,
        )

        # 5) dump
        try:
            self.dump_trd_example(filename="trd_debug.txt", max_steps=50)
        except Exception as e:
            print(f"[TRD] dump_trd_example failed: {e}")

        self.memory.update_index(append=append)

    def record(self, episode: int):
        self.memory.serialize(episode, save_path=self.traces_dir)

    def summary(self):
        self.network.summary()

    def save_weights(self):
        print('saving weights...')
        self.network.save_weights()

    def load_weights(self):
        print('loading weights...')
        self.network.load_weights()

    def save_config(self):
        print('save config')
        self.update_config(policy_lr=self.policy_lr.serialize(), value_lr=self.value_lr.serialize(),
                           adv_scale=self.adv_scale.serialize(),
                           entropy_strength=self.entropy_strength.serialize(), clip_ratio=self.clip_ratio.serialize())
        super().save_config()

    def load_config(self):
        print('load config')
        super().load_config()

        self.policy_lr.load(config=self.config.get('policy_lr', {}))
        self.value_lr.load(config=self.config.get('value_lr', {}))
        self.adv_scale.load(config=self.config.get('adv_scale', {}))
        self.entropy_strength.load(config=self.config.get('entropy_strength', {}))
        self.clip_ratio.load(config=self.config.get('clip_ratio', {}))

    def reset(self):
        super().reset()
        self.network.reset()

    def on_episode_end(self):
        super().on_episode_end()
        self.policy_lr.on_episode()
        self.value_lr.on_episode()
        self.adv_scale.on_episode()
        # ✅ 熵系数/clip_ratio 也允许随 episode 调度
        # 这样训练脚本可以做“先探索、后收敛”的 schedule
        if hasattr(self, "entropy_strength"):
            self.entropy_strength.on_episode()
        if hasattr(self, "clip_ratio"):
            self.clip_ratio.on_episode()

    def on_episode_start(self):
        """Episode initialization for PPO agent."""
        try:
            super().on_episode_start()
            self._log_episode_start()
        except Exception as e:
            logger.error(f"PPO episode start failed: {e}")
            raise

    def _log_episode_start(self):
        msg = "Starting new episode"
        if hasattr(self, 'exploration_rate'):
            msg += f" with exploration rate: {self.exploration_rate:.4f}"
        logger.info(msg)

    def get_exploration_rate(self) -> Optional[float]:
        return getattr(self, 'exploration_rate', None)

    # TRD
    def dump_trd_example(self, filename="trd_debug.txt", max_steps=50):
        if not hasattr(self.memory, "trd_targets") or self.memory.trd_targets is None:
            print("[TRD] no trd_targets in memory, skip dump.")
            return

        try:
            trd = self.memory.trd_targets
            returns = self.memory.returns
            values = self.memory.values

            trd = trd.numpy() if isinstance(trd, tf.Tensor) else np.array(trd)
            returns = returns.numpy() if isinstance(returns, tf.Tensor) else np.array(returns)
            values = values.numpy() if isinstance(values, tf.Tensor) else np.array(values)

            T, dim = trd.shape
            steps = min(T, max_steps)

            V = values[:-1, 0] * np.power(10.0, values[:-1, 1])
            R = returns[:, 0] * np.power(10.0, returns[:, 1])

            filepath = os.path.join(self.base_path, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            print(f"[TRD] dumping to: {filepath}")

            with open(filepath, "a") as f:
                for t in range(steps):
                    sum_trd = float(np.sum(trd[t]))
                    line = {
                        "t": int(t),
                        "V": float(V[t]),
                        "R": float(R[t]),
                        "sum_trd": sum_trd,
                        "trd": trd[t].tolist(),
                    }
                    f.write(json.dumps(line) + "\n")

        except Exception as e:
            print(f"[TRD] dump_trd_example internal error: {e}")


class PPOMemory:
    """Recent memory used in PPOAgent"""

    def __init__(self, state_spec: dict, num_actions: int):
        self.index = 0

        if list(state_spec.keys()) == ['state']:
            self.states = tf.zeros(shape=(0,) + state_spec.get('state'), dtype=tf.float32)
            self.simple_state = True
        else:
            self.states = dict()
            self.simple_state = False
            for name, shape in state_spec.items():
                self.states[name] = tf.zeros(shape=(0,) + shape, dtype=tf.float32)

        self.rewards = tf.zeros(shape=(0,), dtype=tf.float32)
        self.values = tf.zeros(shape=(0, 2), dtype=tf.float32)
        self.actions = tf.zeros(shape=(0, num_actions), dtype=tf.float32)
        # self.log_probabilities = tf.zeros(shape=(0, num_actions), dtype=tf.float32)
        self.log_probabilities = tf.zeros((0,), dtype=tf.float32)  # ✅ 只存 joint logp

        self.timesteps = tf.zeros(shape=(0,), dtype=tf.float32)

        self.returns = None
        self.advantages = None

        self.returns_scalar = None  # [T]

        # TRD
        self.trd_targets = None

    def __len__(self):
        return self.actions.shape[0]

    def delete(self):
        if self.simple_state:
            del self.states
        else:
            for k in self.states.keys():
                self.states[k] = None
            del self.states

        del self.rewards
        del self.values
        del self.actions
        del self.log_probabilities
        del self.timesteps
        del self.returns
        del self.advantages
        del self.trd_targets

    def append(self, state, action, reward, value, log_prob):
        if self.simple_state:
            s = tf.convert_to_tensor(state, dtype=tf.float32)
            if s.shape.rank == len(self.states.shape) - 1:  # 缺 batch
                s = tf.expand_dims(s, axis=0)
            self.states = tf.concat([self.states, s], axis=0)

        else:
            assert isinstance(state, dict)
            for k, v in state.items():
                self.states[k] = tf.concat([self.states[k], v], axis=0)

        a = tf.convert_to_tensor(action, dtype=tf.float32)
        a = tf.reshape(a, [-1, self.actions.shape[1]])  # [-1, A]
        self.actions = tf.concat([self.actions, a], axis=0)

        self.rewards = tf.concat([self.rewards, [reward]], axis=0)
        self.values = tf.concat([self.values, value], axis=0)
        # self.log_probabilities = tf.concat([self.log_probabilities, log_prob], axis=0)
        # ===== log_probabilities: 统一存 joint 标量 =====
        log_prob = tf.convert_to_tensor(log_prob, dtype=tf.float32)

        # 你现在 predict() 返回的 log_prob 可能是 [B,3]（为了日志）
        # PPO 训练需要 joint log_prob: [B]
        if log_prob.shape.rank is None:
            # 极少数情况下静态 rank 不可得，做一个保守 reshape
            log_prob = tf.reshape(log_prob, [tf.shape(log_prob)[0], -1])
            log_prob = tf.reduce_sum(log_prob, axis=-1)
        elif log_prob.shape.rank >= 2:
            # [B, K] -> [B]
            log_prob = tf.reduce_sum(log_prob, axis=-1)

        # 强制成 [B]
        log_prob = tf.reshape(log_prob, [-1])

        # self.log_probabilities 可能是 [0]（空向量），这是 OK 的
        if self.log_probabilities is None:
            self.log_probabilities = log_prob
        else:
            # 强制已有缓存也是 [T]
            self.log_probabilities = tf.reshape(self.log_probabilities, [-1])
            self.log_probabilities = tf.concat([self.log_probabilities, log_prob], axis=0)

    def end_trajectory(self, last_value: tf.Tensor):
        # bootstrap value
        self.values = tf.concat([self.values, last_value], axis=0)
        # dummy reward (for alignment with value bootstrap)
        self.rewards = tf.concat([self.rewards, tf.constant([0.0], dtype=tf.float32)], axis=0)

    def compute_trd_targets(self, discount: float, append: bool = False, n_bins: int = 10):
        """
        生成 TRD targets:
          trd_targets[t, b] = sum_{k in bin(b)} gamma^k * r_{t+k}
        其中 bins 把 [0, T-t) 均匀切成 (n_bins+1) 段
        输出 shape: [T, n_bins+1]
        """
        # rewards 里可能有 dummy，取当前 episode 的环境 reward
        rewards_seq = self.rewards[self.index:]
        # ✅ 去掉 dummy reward，避免越界和统计污染
        rewards_env = rewards_seq[:-1]

        # ✅ 真实步数 T 必须以 rewards_env 长度为准
        # 否则 append=True 时会出现越界（历史 action 比当前 reward 长）
        T = int(rewards_env.shape[0])

        # 转成 numpy 方便写循环（T<=512 完全够用）
        r = rewards_env.numpy() if isinstance(rewards_env, tf.Tensor) else np.asarray(rewards_env, dtype=np.float32)
        r = r.astype(np.float32)

        B = int(n_bins) + 1
        trd = np.zeros((T, B), dtype=np.float32)

        # 逐时刻分解 return-to-go
        for t in range(T):
            rem = T - t  # 剩余可用步数

            # 把 [0, rem] 切成 B 段 => 需要 B+1 个边界
            # 用 round 可能出现重复边界（空 bin），这是允许的，贡献为0
            edges = np.linspace(0, rem, B + 1)
            edges = np.round(edges).astype(int)
            edges[0] = 0
            edges[-1] = rem

            for b in range(B):
                start = edges[b]
                end = edges[b + 1]
                if end <= start:
                    continue

                s = 0.0
                # 注意 k 是相对 t 的 offset，所以折扣是 gamma^k
                for k in range(start, end):
                    s += (discount ** k) * float(r[t + k])
                trd[t, b] = s

        trd_targets = tf.convert_to_tensor(trd, dtype=tf.float32)

        if (self.trd_targets is None) or (not append):
            self.trd_targets = trd_targets
        else:
            self.trd_targets = tf.concat([self.trd_targets, trd_targets], axis=0)

        return self.trd_targets

    def compute_returns(self, discount: float, append=False):
        # ✅ rewards 序列始终包含 dummy（end_trajectory 追加）
        # 这里统一丢掉最后一个 dummy，避免返回长度错位/越界
        rewards_seq = self.rewards[self.index:]
        # ✅ end_trajectory 一定追加 dummy，所以这里直接去掉最后一个
        rewards_env = rewards_seq[:-1]

        returns_scalar = utils.rewards_to_go(rewards_env, discount=discount)
        returns_scalar = utils.to_float(returns_scalar)  # [T]

        new_returns = tf.map_fn(fn=utils.decompose_number, elems=returns_scalar,
                                dtype=(tf.float32, tf.float32))
        new_returns = tf.stack(new_returns, axis=1)  # [T,2]

        if (self.returns is None) or (not append):
            self.returns = new_returns
            self.returns_scalar = returns_scalar
        else:
            self.returns = tf.concat([self.returns, new_returns], axis=0)
            self.returns_scalar = tf.concat([self.returns_scalar, returns_scalar], axis=0)

        return returns_scalar, new_returns

    def compute_advantages(self, gamma: float, lambda_: float, scale=2.0, append=False):
        """
        ✅ 与 utils.gae 的实现严格对齐：
        - rewards: [T+1] (最后一个 dummy)
        - values_full: [T+1] (最后一个 bootstrap)
        输出 advantages: [T]
        """
        rewards = self.rewards[self.index:]  # [T+1]

        # values_full: [T+1]
        values_full = self.values[self.index:, 0] * tf.pow(10.0, self.values[self.index:, 1])

        # --- 强一致性检查（宁可裁剪也不要错位）---
        r_len = tf.shape(rewards)[0]
        v_len = tf.shape(values_full)[0]
        min_len = tf.minimum(r_len, v_len)

        rewards = rewards[:min_len]  # still [T+1]
        values_full = values_full[:min_len]  # still [T+1]

        # 需要至少 2 个点才能算 deltas
        if tf.shape(values_full)[0] < 2:
            adv = tf.zeros((0,), dtype=tf.float32)
            if (self.advantages is None) or (not append):
                self.advantages = adv
            else:
                self.advantages = tf.concat([self.advantages, adv], axis=0)
            return values_full, adv

        advantages = utils.gae(rewards, values=values_full, gamma=gamma, lambda_=lambda_, normalize=False)  # [T]
        # ✅ dtype 保护：gae 内部会走 numpy/scipy，容易变成 float64
        # 这里统一转成 float32，避免后面 PPO 计算报错
        advantages = utils.to_float(advantages)
        # ✅ 只做尺度放大，不在这里做归一化
        # 原来这里 + policy_objective 再标准化会导致“双重归一化”，
        # 使优势过小、梯度变弱，容易卡平台不收敛。
        # 统一在 policy_objective 里做一次标准化即可。
        new_advantages = advantages * scale

        if (self.advantages is None) or (not append):
            self.advantages = new_advantages
        else:
            self.advantages = tf.concat([self.advantages, new_advantages], axis=0)

        return values_full, advantages  # values_full [T+1], advantages [T]

    def update_index(self, append=False):
        """
        ✅ rewards 现在是 [T+1]，最后一个 dummy
        - 如果不 append：直接跳到末尾
        - 如果 append：保留 dummy 作为分界点，index 也跳到末尾（最安全）
        """
        self.index = int(self.rewards.shape[0])

    def serialize(self, episode: int, save_path: str):
        filename = f'trace-{episode}-{time.strftime("%Y%m%d-%H%M%S")}.npz'
        trace_path = os.path.join(save_path, filename)
        buffer = dict(reward=self.rewards, action=self.actions, value=self.values, log_prob=self.log_probabilities)

        if self.simple_state:
            buffer['state'] = self.states
        else:
            for key, value in self.states.items():
                buffer[key] = value

        np.savez_compressed(file=trace_path, **buffer)
        print(f'Traces "{filename}" saved.')
