
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from typing import List, Union, Dict
tfd = tfp.distributions

from .. import utils


class Network:
    def __init__(self, agent):
        self.agent = agent

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def act(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    def trainable_variables(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def load_weights(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError

    def summary(self):
        pass

    def _get_input_layers(self, include_actions=False) -> Dict[str, Input]:
        """Transforms arbitrary complex state-spaces (and, optionally, action-spaces) as input layers"""
        input_layers = dict()

        for name, shape in self.agent.state_spec.items():
            # if self.agent.drop_batch_remainder:
            #     layer = Input(shape=shape, batch_size=self.agent.batch_size, dtype=tf.float32, name=name)
            # else:
            #     layer = Input(shape=shape, dtype=tf.float32, name=name)

            layer = Input(shape=shape, dtype=tf.float32, name=name)
            input_layers[name] = layer

        # TODO: bugged for discrete actions (requires '-1' in shape)
        if include_actions:
            for name, shape in self.agent.action_spec.items():
                layer = Input(shape=shape, dtype=tf.float32, name=name)
                input_layers[name] = layer

        return input_layers

    @staticmethod
    def _clip_actions(actions):
        """Clips actions to prevent numerical instability when computing (log-)probabilities.
           - Use for Beta distribution only.
        """
        return tf.clip_by_value(actions, utils.EPSILON, 1.0 - utils.EPSILON)

    def get_distribution_layer(self, distribution: str, layer: Layer) -> tfp.layers.DistributionLambda:
        # Discrete actions:
        if distribution == 'categorical':
            num_actions = self.agent.num_actions
            num_classes = self.agent.num_classes

            logits = Dense(units=num_actions * num_classes, activation='linear', name='logits')(layer)

            if num_actions > 1:
                logits = Reshape((num_actions, num_classes))(logits)
            else:
                logits = tf.expand_dims(logits, axis=0)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Categorical(logits=t))(logits)

        # Bounded continuous 1-dimensional actions:
        # for activations choice refer to chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if distribution == 'beta':
            num_actions = self.agent.num_actions

            # make a, b > 1, so that the Beta distribution is concave and unimodal (see paper above)
            alpha = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='alpha')(layer)
            beta = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='beta')(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Beta(t[0], t[1]))([alpha, beta])

        # Unbounded continuous actions)
        # for activations choice see chapter 4 of http://proceedings.mlr.press/v70/chou17a/chou17a.pdf
        if distribution == 'gaussian':
            num_actions = self.agent.num_actions

            mu = Dense(units=num_actions, activation='linear', name='mu')(layer)
            sigma = Dense(units=num_actions, activation=utils.softplus(1.0 + 1e-2), name='sigma')(layer)

            return tfp.layers.DistributionLambda(
                make_distribution_fn=lambda t: tfp.distributions.Normal(loc=t[0], scale=t[1]))([mu, sigma])


# TODO: disentangle policy-net from value-net, so that each of them can be arbitrary subclassed, moreover a
#  Network class can be composed by these policy/value/Q-network classes...
# Network class can be composed by these policy/value/Q-network classes...
class PPONetwork(Network):
    def __init__(self, agent, policy: dict, value: dict):
        from ..agents.ppo import PPOAgent
        super().__init__(agent)
        self.agent: PPOAgent

        self.distribution = self.agent.distribution_type

        # TRD (设置在创建网络之前,因为value_network需要用到)
        self.trd_bins = getattr(self.agent, "trd_bins", 10)

        # ====== NEW: two-stage policy ======
        # policy_lat: dist over [steer, y_ref]
        self.policy_lat = self.policy_lat_network(**policy)
        self.old_policy_lat = self.policy_lat_network(**policy)

        # policy_lon: dist over [throttle], conditioned on y_ref
        self.policy_lon = self.policy_lon_network(**policy)
        self.old_policy_lon = self.policy_lon_network(**policy)

        # 同步 old
        self.update_old_policy()

        # value network (输出: [value(base,exp), trd_vec])
        self.exp_scale = 6.0
        self.value = self.value_network(**value)

        # last_value: (base, exp)
        self.last_value = tf.zeros((1, 2), dtype=tf.float32)

    # ----------------- small utils -----------------
    def _as_input_list(self, inputs):
        if isinstance(inputs, dict):
            return list(inputs.values())
        return [inputs]

    def _unwrap_distribution(self, dist):
        """
        Keras/TFP 的 DistributionLambda 可能返回带包装器的分布对象。
        这里尽量剥离到真实分布，避免后续 isinstance 判断失效。
        """
        d = dist
        # 最多展开几层，避免异常对象循环引用
        for _ in range(6):
            inner = getattr(d, "distribution", None)
            if inner is None or inner is d:
                break
            d = inner
        return d

    def _get_base_normal(self, dist):
        """
        dist 可能是 Normal / TransformedDistribution(Normal, ...) / Independent(...) / DistributionLambda返回的对象
        目标：返回一个有 .loc / .scale 的分布（通常是 Normal）
        """
        d = self._unwrap_distribution(dist)

        # 逐层下钻到 base distribution（通常是 Normal）
        for _ in range(6):
            if isinstance(d, tfd.TransformedDistribution):
                d = d.distribution
                continue

            if isinstance(d, tfd.Independent):
                d = d.distribution
                continue

            inner = getattr(d, "distribution", None)
            if inner is not None and inner is not d:
                d = inner
                continue
            break

        # 兜底：此时 d 应该就是 Normal（至少要有 loc/scale）
        return d

    def _deterministic_from_dist(self, dist):
        """
        对 tanh-squash 分布给出 deterministic 动作。
        - TransformedDistribution(Tanh(Normal))：使用 tanh(loc)
        - 其他分布：优先尝试 mean()，失败则回退到 base loc
        """
        d = self._unwrap_distribution(dist)
        base = self._get_base_normal(d)

        # 优先使用 base.loc（最稳定，可规避 non-affine bijector 的 mean() 未实现问题）
        if hasattr(base, "loc"):
            loc = tf.convert_to_tensor(base.loc, dtype=tf.float32)
            bijector = getattr(d, "bijector", None)
            if bijector is not None:
                try:
                    return tf.cast(bijector.forward(loc), tf.float32)
                except Exception:
                    pass
            return tf.cast(loc, tf.float32)

        # 退化路径：无法拿到 loc 时再尝试 mean/sample
        try:
            return tf.cast(d.mean(), tf.float32)
        except Exception:
            return tf.cast(d.sample(), tf.float32)

    def _infer_batch_size_from_inputs(self, inputs):
        if isinstance(inputs, dict):
            first = list(inputs.values())[0]
            return tf.shape(first)[0]
        return tf.shape(inputs)[0]

    def _policy_weight_paths(self):
        """
        兼容旧版 agent.weights_path['policy'] 只有一个路径的情况：
        - policy_lat 保存为 <policy_path>_lat
        - policy_lon 保存为 <policy_path>_lon
        """
        base = self.agent.weights_path.get("policy", None)
        if base is None:
            base = "policy_net"
        return base + "_lat", base + "_lon"

    # ----------------- logprob & entropy -----------------
    @tf.function
    def log_prob_and_entropy(self, states, actions, training: bool = True, use_old: bool = False):
        """
        actions: [B,3] = [throttle_brake, steer, y_ref]
        return:
          log_prob_joint: [B]
          entropy: scalar (batch mean, Monte-Carlo approx)
        """
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        # ✅ 关键：永远强制成 [B,3]，杜绝 rank/shape 分支导致变量未赋值
        actions = tf.reshape(actions, [-1, 3])

        eps = tf.constant(1e-6, dtype=tf.float32)

        # ✅ 无条件赋值
        throttle_brake = actions[:, 0:1]
        steer = actions[:, 1:2]
        y_ref = actions[:, 2:3]

        # ✅ clip 防止 log_prob 数值炸
        throttle_brake = tf.clip_by_value(throttle_brake, -1.0 + eps, 1.0 - eps)
        steer = tf.clip_by_value(steer, -1.0 + eps, 1.0 - eps)
        y_ref = tf.clip_by_value(y_ref, -1.0 + eps, 1.0 - eps)

        lat_net = self.old_policy_lat if use_old else self.policy_lat
        lon_net = self.old_policy_lon if use_old else self.policy_lon

        # ========== LAT: p([steer, y_ref] | s) ==========
        dist_lat = lat_net(states, training=training)
        lat_act = tf.concat([steer, y_ref], axis=1)  # [B,2]

        lat_logp = dist_lat.log_prob(lat_act)
        if tf.rank(lat_logp) > 1:
            lat_logp = tf.reduce_sum(lat_logp, axis=-1)  # [B]

        # ✅ entropy：TransformedDistribution.entropy() 常常不实现，用 MC 近似
        lat_sample = dist_lat.sample()  # [B,2]
        lat_logp_s = dist_lat.log_prob(lat_sample)
        if tf.rank(lat_logp_s) > 1:
            lat_logp_s = tf.reduce_sum(lat_logp_s, axis=-1)
        lat_ent = -lat_logp_s  # [B]

        # ========== LON: p(throttle_brake | s, y_ref) ==========
        inputs_list = self._as_input_list(states)
        if isinstance(states, dict):
            dist_lon = lon_net({**states, "y_ref": y_ref}, training=training)
        else:
            dist_lon = lon_net([states, y_ref], training=training)

        lon_logp = dist_lon.log_prob(throttle_brake)
        if tf.rank(lon_logp) > 1:
            lon_logp = tf.reduce_sum(lon_logp, axis=-1)  # [B]

        lon_sample = dist_lon.sample()  # [B,1]
        lon_logp_s = dist_lon.log_prob(lon_sample)
        if tf.rank(lon_logp_s) > 1:
            lon_logp_s = tf.reduce_sum(lon_logp_s, axis=-1)
        lon_ent = -lon_logp_s  # [B]

        log_prob_joint = lon_logp + lat_logp  # [B]
        entropy = tf.reduce_mean(lon_ent + lat_ent)  # scalar

        return log_prob_joint, entropy

    @tf.function
    def act(self, inputs, use_old: bool = True, deterministic: bool = False):
        lat_net = self.old_policy_lat if use_old else self.policy_lat
        lon_net = self.old_policy_lon if use_old else self.policy_lon

        dist_lat = lat_net(inputs, training=False)

        if deterministic:
            lat = self._deterministic_from_dist(dist_lat)
        else:
            lat = dist_lat.sample()

        steer = lat[:, 0:1]
        y_ref = lat[:, 1:2]

        inputs_list = self._as_input_list(inputs)
        dist_lon = lon_net(inputs_list + [y_ref], training=False)

        if deterministic:
            throttle = self._deterministic_from_dist(dist_lon)
        else:
            throttle = dist_lon.sample()

        return tf.concat([throttle, steer, y_ref], axis=1)

    @tf.function
    def act2(self, inputs, use_old: bool = True, deterministic: bool = False):
        """
        返回：(action, log_prob, value)
        - log_prob 一定与采样策略一致（use_old 同步）
        """
        action = self.act(inputs, use_old=use_old, deterministic=deterministic)
        value = self.value_predict(inputs)
        log_prob, _ = self.log_prob_and_entropy(inputs, action, training=False, use_old=use_old)
        return action, log_prob, value

    # ----------------- inference: predict -----------------
    # ----------------- inference: predict -----------------
    @tf.function
    def predict(self, inputs):
        """
        返回：
          action: [B,3] = [throttle, steer, y_ref]
          mean/std/log_prob: [B,3]（用于日志可视化）
          value: [B,2]
        """
        eps = tf.constant(1e-6, dtype=tf.float32)

        # 0) value
        value = self.value_predict(inputs)

        # 1) 横向 leader：dist over [steer, y_ref]
        dist_lat = self.old_policy_lat(inputs, training=False)

        # ✅ 必须：先赋值再 clip（避免 UnboundLocalError）
        # 如果你有 deterministic 开关，就在这里统一处理
        if hasattr(self, "deterministic") and bool(self.deterministic):
            lat_sample = self._deterministic_from_dist(dist_lat)  # [B,2] in (-1,1) for tanh dist
        else:
            lat_sample = dist_lat.sample()  # [B,2]

        lat_sample = tf.clip_by_value(lat_sample, -1.0 + eps, 1.0 - eps)

        # logp（可能是 [B] 或 [B,2]）
        lat_logp = dist_lat.log_prob(lat_sample)

        # mean/std（可视化用近似）
        base_lat = self._get_base_normal(dist_lat)
        lat = tf.tanh(base_lat.loc)
        lat_mean_raw = base_lat.loc
        lat_std_raw = base_lat.scale

        # mean 用 tanh 映射只是为了“日志可视化近似”
        lat_mean = tf.tanh(lat_mean_raw)
        lat_std = lat_std_raw

        steer = lat_sample[:, 0:1]
        y_ref = lat_sample[:, 1:2]

        # 2) 纵向 follower：dist over [throttle] conditioned on y_ref
        inputs_list = self._as_input_list(inputs)
        dist_lon = self.old_policy_lon(inputs_list + [y_ref], training=False)

        if hasattr(self, "deterministic") and bool(self.deterministic):
            lon_sample = self._deterministic_from_dist(dist_lon)  # [B,1]
        else:
            lon_sample = dist_lon.sample()  # [B,1]

        lon_sample = tf.clip_by_value(lon_sample, -1.0 + eps, 1.0 - eps)
        lon_logp = dist_lon.log_prob(lon_sample)

        base_lon = self._get_base_normal(dist_lon)
        # ✅ 关键修复：执行动作必须与 log_prob 对齐
        # 原来这里用的是 tanh(mu)，但 log_prob 用的是采样值 lon_sample，
        # 会导致 PPO 比率失真，训练容易卡平台/不收敛。
        # 这里改为：动作使用采样值（与 log_prob 一致），
        # 均值仅用于日志可视化。
        throttle_action = lon_sample
        lon_mean_raw = base_lon.loc
        lon_std_raw = base_lon.scale
        lon_mean = tf.tanh(lon_mean_raw)
        lon_std = lon_std_raw


        # 3) 拼最终 action / mean / std / log_prob
        action = tf.concat([throttle_action, steer, y_ref], axis=1)  # [B,3]
        mean = tf.concat([lon_mean, lat_mean[:, 0:1], lat_mean[:, 1:2]], axis=1)
        std = tf.concat([lon_std, lat_std[:, 0:1], lat_std[:, 1:2]], axis=1)

        # 计算 joint logp（用于 PPO）
        # lat_logp_vec: [B]
        lat_logp_vec = dist_lat.log_prob(lat_sample)
        if tf.rank(lat_logp_vec) > 1:
            lat_logp_vec = tf.reduce_sum(lat_logp_vec, axis=-1)

        lon_logp_vec = dist_lon.log_prob(lon_sample)
        if tf.rank(lon_logp_vec) > 1:
            lon_logp_vec = tf.reduce_sum(lon_logp_vec, axis=-1)

        log_prob_joint = lon_logp_vec + lat_logp_vec  # [B]

        return action, mean, std, log_prob_joint, value

    # ----------------- policy/value predict -----------------
    def policy_predict(self, inputs):
        """
        原来是 return self.policy(...) —— 现在没有 self.policy 了。
        这里返回两个分布，方便 debug：
        - dist_lat(s)
        - dist_lon(s, y_ref=0) 只是占位推一下（不用于训练）
        """
        dist_lat = self.policy_lat(inputs, training=False)
        batch = self._infer_batch_size_from_inputs(inputs)
        y0 = tf.zeros((batch, 1), dtype=tf.float32)
        dist_lon = self.policy_lon(self._as_input_list(inputs) + [y0], training=False)
        return dist_lat, dist_lon

    @tf.function
    def value_predict(self, inputs):
        value, _ = self.value(inputs, training=False)
        return value

    @tf.function
    def value_and_trd(self, inputs, training: bool = False):
        value, trd_vec = self.value(inputs, training=training)
        return value, trd_vec

    def predict_last_value(self, state, timestep: float, is_terminal: bool):
        if is_terminal:
            return self.last_value
        return self.value_predict(state)

    # ----------------- MLP blocks -----------------
    def policy_layers(self, inputs: Dict[str, Input], **kwargs):
        units = kwargs.get('units', 32)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))
        activation = kwargs.get('activation', tf.nn.swish)
        dropout_rate = kwargs.get('dropout', 0.0)
        linear_units = kwargs.get('linear_units', 0)

        x = Dense(units, activation=activation)(inputs['state'])
        x = LayerNormalization()(x)

        for _ in range(0, num_layers, 2):
            if dropout_rate > 0.0:
                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)

                x = Dense(units, activation=activation)(x)
                x = Dropout(rate=dropout_rate)(x)
            else:
                x = Dense(units, activation=activation)(x)
                x = Dense(units, activation=activation)(x)

            x = LayerNormalization()(x)

        if linear_units > 0:
            x = Dense(units=linear_units, activation='linear')(x)

        return x

    def value_layers(self, inputs: Dict[str, Input], **kwargs):
        return self.policy_layers(inputs, **kwargs)

    # ----------------- dist builders -----------------
    def _squashed_gaussian_dist_layer(
        self,
        layer,
        out_dim,
        name_prefix,
        min_scale=0.02,
        raw_scale_bias=-1.2,
        raw_scale_clip_min=-4.0,
        raw_scale_clip_max=-0.1,
    ):
        mu = Dense(out_dim, activation='linear', name=f'{name_prefix}_mu')(layer)

        # raw_scale -> softplus -> scale
        # 关键：初始化和上限都更保守，避免 std 长期卡在 ~1.0 导致动作抖动。
        raw_scale = Dense(out_dim,
                          activation='linear',
                          kernel_initializer='zeros',
                          bias_initializer=tf.keras.initializers.Constant(raw_scale_bias),
                          name=f'{name_prefix}_raw_scale')(layer)
        raw_scale = tf.clip_by_value(raw_scale, raw_scale_clip_min, raw_scale_clip_max)
        scale = tf.nn.softplus(raw_scale) + min_scale

        def _make_dist(params):
            mu_, scale_ = params
            base = tfp.distributions.Normal(loc=mu_, scale=scale_)

            # tanh squash 到 (-1,1)
            dist = tfp.distributions.TransformedDistribution(
                distribution=base,
                bijector=tfp.bijectors.Tanh()
            )
            return dist

        return tfp.layers.DistributionLambda(_make_dist)([mu, scale])

    def policy_lat_network(self, **kwargs):
        """
        输入 state，输出二维 Normal 分布：[steer, y_ref]
        """
        inputs = self._get_input_layers()
        last_layer = self.policy_layers(inputs, **kwargs)
        dist_lat = self._squashed_gaussian_dist_layer(
            last_layer,
            out_dim=2,
            name_prefix='lat',
            min_scale=float(kwargs.get("min_scale_lat", kwargs.get("min_scale", 0.02))),
            raw_scale_bias=float(kwargs.get("raw_scale_bias_lat", kwargs.get("raw_scale_bias", -1.2))),
            raw_scale_clip_min=float(kwargs.get("raw_scale_clip_min_lat", kwargs.get("raw_scale_clip_min", -4.0))),
            raw_scale_clip_max=float(kwargs.get("raw_scale_clip_max_lat", kwargs.get("raw_scale_clip_max", -0.1))),
        )

        return Model(list(inputs.values()), outputs=dist_lat, name='Policy-Lat')

    def policy_lon_network(self, **kwargs):
        """
        输入 state + y_ref，输出一维 Normal 分布：[throttle]
        """
        inputs = self._get_input_layers()
        y_ref_in = Input(shape=(1,), dtype=tf.float32, name='y_ref')

        x = concatenate([inputs['state'], y_ref_in], axis=1)

        units = kwargs.get('units', 32)
        num_layers = kwargs.get('num_layers', kwargs.get('layers', 2))
        activation = kwargs.get('activation', tf.nn.swish)

        x = Dense(units, activation=activation)(x)
        x = LayerNormalization()(x)
        for _ in range(0, num_layers, 2):
            x = Dense(units, activation=activation)(x)
            x = Dense(units, activation=activation)(x)
            x = LayerNormalization()(x)

        dist_lon = self._squashed_gaussian_dist_layer(
            x,
            out_dim=1,
            name_prefix='lon',
            min_scale=float(kwargs.get("min_scale_lon", kwargs.get("min_scale", 0.02))),
            raw_scale_bias=float(kwargs.get("raw_scale_bias_lon", kwargs.get("raw_scale_bias", -1.2))),
            raw_scale_clip_min=float(kwargs.get("raw_scale_clip_min_lon", kwargs.get("raw_scale_clip_min", -4.0))),
            raw_scale_clip_max=float(kwargs.get("raw_scale_clip_max_lon", kwargs.get("raw_scale_clip_max", -0.1))),
        )

        return Model(list(inputs.values()) + [y_ref_in], outputs=dist_lon, name='Policy-Lon')

    # ----------------- value network (+TRD) -----------------
    def value_network(self, **kwargs):
        inputs = self._get_input_layers()
        last_layer = self.value_layers(inputs, **kwargs)

        value = self.value_head(last_layer, **kwargs)
        self.exp_scale = kwargs.get('exponent_scale', self.exp_scale)

        trd_units = self.trd_bins + 1
        trd_vec = Dense(units=trd_units, activation='linear', name='trd-head')(last_layer)

        return Model(list(inputs.values()), outputs=[value, trd_vec], name='Value-Network')

    def value_head(self, layer: Layer, exponent_scale=6.0, components=1, **kwargs):
        assert components >= 1
        assert exponent_scale > 0.0

        if components == 1:
            base = Dense(units=1, activation=tf.nn.tanh, name='v-base')(layer)
            exp = Dense(units=1, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), name='v-exp')(layer)
        else:
            weights_base = Dense(units=components, activation='softmax', name='w-base')(layer)
            weights_exp = Dense(units=components, activation='softmax', name='w-exp')(layer)

            base = Dense(units=components, activation=tf.nn.tanh, name='v-base')(layer)
            base = utils.tf_dot_product(base, weights_base, axis=1, keepdims=True)

            exp = Dense(units=components, activation=lambda x: exponent_scale * tf.nn.sigmoid(x), name='v-exp')(layer)
            exp = utils.tf_dot_product(exp, weights_exp, axis=1, keepdims=True)

        return concatenate([base, exp], axis=1)

    # ----------------- weights I/O -----------------
    def load_weights(self):
        lat_path, lon_path = self._policy_weight_paths()
        val_path = self.agent.weights_path.get('value', 'value_net')

        self.policy_lat.load_weights(filepath=lat_path)
        self.policy_lon.load_weights(filepath=lon_path)
        self.value.load_weights(filepath=val_path)

        self.update_old_policy()  # ✅ 同步 old_policy_lat/lon

        print(f"[PPONetwork] ✅ loaded policy_lat from: {lat_path}")
        print(f"[PPONetwork] ✅ loaded policy_lon from: {lon_path}")
        print(f"[PPONetwork] ✅ loaded value      from: {val_path}")

    def save_weights(self):
        lat_path, lon_path = self._policy_weight_paths()
        val_path = self.agent.weights_path.get('value', 'value_net')

        self.policy_lat.save_weights(filepath=lat_path)
        self.policy_lon.save_weights(filepath=lon_path)
        self.value.save_weights(filepath=val_path)

        print(f"[PPONetwork] ✅ saved policy_lat to: {lat_path}")
        print(f"[PPONetwork] ✅ saved policy_lon to: {lon_path}")
        print(f"[PPONetwork] ✅ saved value     to: {val_path}")

    def update_old_policy(self, weights=None):
        self.old_policy_lat.set_weights(self.policy_lat.get_weights())
        self.old_policy_lon.set_weights(self.policy_lon.get_weights())

    def summary(self):
        print('==== Policy Lat (steer,y_ref) ====')
        self.policy_lat.summary()
        print('\n==== Policy Lon (throttle | y_ref) ====')
        self.policy_lon.summary()
        print('\n==== Value Network (+TRD) ====')
        self.value.summary()
