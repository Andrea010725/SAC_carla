import os
import gym
import math

import ipdb
import numpy as np
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from typing import Union, List, Dict, Tuple, Optional
from distutils import dir_util
from datetime import datetime

from gym import spaces

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from .parameters import DynamicParameter
import ipdb


# -------------------------------------------------------------------------------------------------
# -- Constants
# -------------------------------------------------------------------------------------------------

NP_EPS = np.finfo(np.float32).eps
EPSILON = tf.constant(NP_EPS, dtype=tf.float32)

TF_ZERO = tf.constant(0.0, dtype=tf.float32)

OPTIMIZERS = dict(adadelta=tf.keras.optimizers.Adadelta,
                  adagrad=tf.keras.optimizers.Adagrad,
                  adam=tf.keras.optimizers.Adam,
                  adamax=tf.keras.optimizers.Adamax,
                  ftrl=tf.keras.optimizers.Ftrl,
                  nadam=tf.keras.optimizers.Nadam,
                  rmsprop=tf.keras.optimizers.RMSprop,
                  sgd=tf.keras.optimizers.SGD)


def get_optimizer_by_name(name: str, *args, **kwargs) -> tf.keras.optimizers.Optimizer:
    optimizer_class = OPTIMIZERS.get(name.lower(), None)

    if optimizer_class is None:
        raise ValueError(f'Cannot find optimizer {name}. Select one of {OPTIMIZERS.keys()}.')

    print(f'Optimizer: {name}.')
    return optimizer_class(*args, **kwargs)


# -------------------------------------------------------------------------------------------------
# -- Misc
# -------------------------------------------------------------------------------------------------

def np_normalize(x, epsilon=np.finfo(np.float32).eps):
    return (x - np.mean(x)) / (np.std(x) + epsilon)


def discount_cumsum(x, discount: float):
    """Source: https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/core.py#L45"""
    return scipy.signal.lfilter([1.0], [1.0, float(-discount)], x[::-1], axis=0)[::-1]


def gae(rewards, values, gamma: float, lambda_: float, normalize=False):
    if lambda_ == 0.0:
        advantages = rewards[:-1] + gamma * values[1:] - values[:-1]
    else:
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        advantages = discount_cumsum(deltas, discount=gamma * lambda_)

    if normalize:
        advantages = tf_normalize(advantages)

    return advantages


def rewards_to_go(rewards, discount: float, decompose=False, has_dummy: bool = False):
    """
    rewards:
      - has_dummy=False: rewards shape [T] (真实环境 reward)
      - has_dummy=True : rewards shape [T+1] (末尾 dummy reward)
    returns:
      - [T]
    """
    rewards = to_float(rewards)
    rtg = discount_cumsum(rewards, discount=discount)

    returns = rtg[:-1] if has_dummy else rtg  # ✅ 关键改动：不再无脑[:-1]

    if decompose:
        returns_base, returns_exp = tf.map_fn(
            fn=decompose_number,
            elems=to_float(returns),
            dtype=(tf.float32, tf.float32)
        )
        return tf.stack([returns_base, returns_exp], axis=1), returns

    return returns



def is_image(x) -> bool:
    """Checks whether some input [x] has a shape of the form (H, W, C)"""
    return len(x.shape) == 3


def is_vector(x) -> bool:
    """Checks whether some input [x] has a shape of the form (N, D) or (D,)"""
    return 1 <= len(x.shape) <= 2


def depth_concat(*arrays):
    return np.concatenate(*arrays, axis=-1)


def clip(value, min_value, max_value):
    return min(max_value, max(value, min_value))


def polyak_averaging(model: tf.keras.Model, old_weights: list, alpha=0.99):
    """Source: Deep Learning Book (section 8.7.3)
        - the original implementation is: `w = alpha * w_old + (1.0 - alpha) * w_new`,
          here we use `w = alpha * w_new + (1.0 - alpha) * w_old` because it performs better for RL
    """
    new_weights = model.get_weights()
    weights = []

    for w_old, w_new in zip(old_weights, new_weights):
        w = alpha * w_new + (1.0 - alpha) * w_old
        weights.append(w)

    model.set_weights(weights)


def clip_gradients(gradients: list, norm: float) -> list:
    out = []
    for g in gradients:
        if g is None:
            out.append(None)
        else:
            out.append(tf.clip_by_norm(g, clip_norm=norm))
    return out



def accumulate_gradients(grads1: list, grads2: Optional[list] = None) -> list:
    if grads2 is None:
        return grads1

    return [g1 + g2 for g1, g2 in zip(grads1, grads2)]


def average_gradients(gradients: list, n: int) -> list:
    assert n > 0
    if n == 1:
        return gradients

    n = float(n)
    return [g / n for g in gradients]


def decompose_number(num: float) -> (float, float):
    """Decomposes a given number [n] in a scientific-like notation:
       - n = fractional_part * 10^exponent
       - e.g. 2.34 could be represented as (0.234, 1) such that 0.234 * 10^1 = 2.34
    """
    # �� NaN防护：检查输入
    if np.isnan(num) or np.isinf(num):
        # tf.print("⚠️  decompose_number: 输入是NaN/Inf，返回(0.0, 0.0)")
        return 0.0, 0.0

    # �� NaN防护：限制exponent范围，避免溢出
    exponent = 0
    max_exponent = 30  # 10^30 已经是非常大的数

    while abs(num) > 1.0 and exponent < max_exponent:
        num /= 10.0
        exponent += 1

    # 如果数字太小，也做类似处理
    min_exponent = -30
    while abs(num) < 0.1 and num != 0.0 and exponent > min_exponent:
        num *= 10.0
        exponent -= 1

    return num, float(exponent)


# -------------------------------------------------------------------------------------------------
# -- Plot utils
# -------------------------------------------------------------------------------------------------

def plot_images(images: list):
    """Plots a list of images, arranging them in a rectangular fashion"""
    num_plots = len(images)
    rows = round(math.sqrt(num_plots))
    cols = math.ceil(math.sqrt(num_plots))

    for k, img in enumerate(images):
        plt.subplot(rows, cols, k + 1)
        plt.axis('off')
        plt.imshow(img)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_lr_schedule(lr_schedule: Union[DynamicParameter, LearningRateSchedule], iterations: int, initial_step=0,
                     show=True):
    assert iterations > 0
    lr_schedule = DynamicParameter.create(value=lr_schedule)

    data = [lr_schedule(step=i + initial_step) for i in range(iterations)]
    plt.plot(data)

    if show:
        plt.show()


# -------------------------------------------------------------------------------------------------
# -- Gym utils
# -------------------------------------------------------------------------------------------------

def print_info(gym_env):
    if isinstance(gym_env, str):
        gym_env = gym.make(gym_env)

    obs_space = gym_env.observation_space
    act_space = gym_env.action_space

    # Observation space:
    if isinstance(obs_space, gym.spaces.Box):
        print(f'Observation space: {obs_space}, shape: {obs_space.shape}, bounds: {obs_space.low}, {obs_space.high}')
    else:
        print(f'Observation space: {obs_space}, n: {obs_space.n}')

    # Action space:
    if isinstance(act_space, gym.spaces.Box):
        print(f'Action space: {act_space}, shape: {act_space.shape}, bounds: {act_space.low}, {act_space.high}')
    else:
        print(f'Action space: {act_space}, n: {act_space.n}')

    print('Reward range:', gym_env.reward_range)
    print('Metadata:', gym_env.metadata)


def space_to_flat_spec(space: gym.Space, name: str) -> Dict[str, tuple]:
    """From a gym.Space object returns a flat dictionary str -> tuple.
       Naming convention:
         - If space is Box or Discrete, it returns 'dict(name=shape)'
         - If space is Dict (not nested), it returns 'dict(name_x=shape_x, name_y=shape_y)'
            considering 'x' and 'y' be component of space.
         - With further nesting, dict keys' names got created using the above two rules.
           In this way each key (name) uniquely identifies a (sub-)component of the space.
           Example:
              Dict(a=x, b=Dict(c=y, d=z)) -> dict(a=x, b_c=y, b_d=z)
    """
    spec = dict()

    if isinstance(space, spaces.Discrete):
        spec[name] = (space.n,)

    elif isinstance(space, spaces.MultiDiscrete):
        spec[name] = space.nvec.shape

    elif isinstance(space, spaces.Box):
        spec[name] = space.shape

    elif isinstance(space, spaces.Dict):
        for key, value in space.spaces.items():
            space_name = f'{name}_{key}'
            result = space_to_flat_spec(space=value, name=space_name)

            if isinstance(result, dict):
                for k, v in result.items():
                    spec[k] = v
            else:
                spec[space_name] = result
    else:
        raise ValueError('space must be one of Box, Discrete, MultiDiscrete, or Dict')

    return spec


def space_to_spec(space: gym.Space) -> Union[tuple, Dict[str, Union[tuple, dict]]]:
    """From a gym.Space object returns its shape-specification, i.e.
         - tuple: if space is Box or Discrete
         - dict[str, tuple or dict]: if space is spaces.Dict
    """
    if isinstance(space, spaces.Box):
        return space.shape

    if isinstance(space, spaces.Discrete):
        return space.n,  # -> tuple (space.n,)

    if isinstance(space, spaces.MultiDiscrete):
        return space.nvec.shape

    assert isinstance(space, spaces.Dict)

    spec = dict()
    for name, space in space.spaces.items():
        spec[name] = space_to_spec(space)

    return spec


# -------------------------------------------------------------------------------------------------
# -- TF utils
# -------------------------------------------------------------------------------------------------

def to_tensor(x, expand_axis=0):
    if isinstance(x, dict):
        t = dict()

        for k, v in x.items():
            v = to_float(v)
            t[k] = tf.expand_dims(tf.convert_to_tensor(v), axis=expand_axis)

        return t
    else:
        x = to_float(x)
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, axis=expand_axis)

        return x


def tf_replace_nan(tensor, value=0.0, dtype=tf.float32):
    replacement = tf.constant(value, dtype=dtype, shape=tensor.shape)
    return tf.where(tensor == tensor, x=tensor, y=replacement)


def num_dims(tensor) -> tf.int32:
    """Returns the dimensionality (number of dimensions/axis) of the given tensor"""
    return tf.rank(tf.shape(tensor))


def mask_dict_tensor(tensor: dict, mask) -> dict:
    return {k: v[mask] for k, v in tensor.items()}


def concat_tensors(*tensors, axis=0) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
    assert len(tensors) > 0

    if isinstance(tensors[0], dict):
        return concat_dict_tensor(*tensors, axis=axis)

    return tf.concat(tensors, axis=axis)


def concat_dict_tensor(*dicts, axis=0) -> dict:
    assert len(dicts) > 0
    assert isinstance(dicts[0], dict)

    result = dicts[0]

    for i in range(1, len(dicts)):
        d = dicts[i]
        result = {k: tf.concat([v, d[k]], axis=axis) for k, v in result.items()}

    return result


def tf_chance(seed=None):
    """Use to get a single random number between 0 and 1"""
    return tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0, seed=seed)


def tf_normalize(x, eps=EPSILON):
    """Normalizes some tensor x to 0-mean 1-stddev"""
    x = to_float(x)
    return (x - tf.math.reduce_mean(x)) / (tf.math.reduce_std(x) + eps)


def tf_sp_norm(x, eps=1e-3):
    x = to_float(x)

    # �� NaN防护：检查输入
    if tf.reduce_any(tf.math.is_nan(x)) or tf.reduce_any(tf.math.is_inf(x)):
        tf.print("⚠️  tf_sp_norm: 检测到NaN/Inf输入，返回零向量")
        return tf.zeros_like(x)

    positives = x * to_float(x > 0.0)
    negatives = x * to_float(x < 0.0)

    # �� NaN防护：避免除以零
    max_val = tf.reduce_max(x)
    min_val = tf.reduce_min(x)

    if tf.abs(max_val) < eps and tf.abs(min_val) < eps:
        return x

    result = (positives / (max_val + eps)) + (negatives / -(min_val - eps))

    # �� NaN防护：检查输出
    if tf.reduce_any(tf.math.is_nan(result)) or tf.reduce_any(tf.math.is_inf(result)):
        tf.print("⚠️  tf_sp_norm: 输出包含NaN/Inf，返回零向量")
        return tf.zeros_like(x)

    return result


def tf_shuffle_tensors(*tensors, indices=None):
    """Shuffles all the given tensors in the SAME way.
       Source: https://stackoverflow.com/questions/56575877/shuffling-two-tensors-in-the-same-order
    """
    assert len(tensors) > 0

    if indices is None:
        indices = tf.range(start=0, limit=tf.shape(tensors[0])[0], dtype=tf.int32)
        indices = tf.random.shuffle(indices)

    return [tf.gather(t, indices) for t in tensors]


# ============================================================
# ✅ 核心修复：对齐 from_tensor_slices 的第0维长度
# ============================================================

def _first_dim_len(x) -> int:
    """
    返回 x 的第0维长度（dict 则取所有 value 的最小值），用于对齐 dataset 输入。
    这里默认训练是 eager（你现在就是 eager + wandb），因此允许 .numpy()。
    """
    if isinstance(x, dict):
        if len(x) == 0:
            return 0
        return min(_first_dim_len(v) for v in x.values())

    if tf.is_tensor(x):
        if x.shape.rank is not None and x.shape.rank >= 1 and x.shape[0] is not None:
            return int(x.shape[0])
        # 动态 shape（eager）
        return int(tf.shape(x)[0].numpy())

    if isinstance(x, np.ndarray):
        return int(x.shape[0])

    # list/tuple 或其他序列
    if hasattr(x, '__len__'):
        return int(len(x))

    return 0


def _slice_first_dim(x, n: int):
    """把 x 按第0维裁剪到 n（dict 递归裁剪）。"""
    if isinstance(x, dict):
        return {k: _slice_first_dim(v, n) for k, v in x.items()}

    # tf.Tensor / np.ndarray / list 均支持 [:n]
    try:
        return x[:n]
    except Exception:
        return x


def _align_tensors_first_dim(tensors):
    """
    tensors 可以是 dict / (list|tuple) / 单个 tensor
    统一裁到共同最小长度 n，避免 from_tensor_slices 报 Dimensions ... not compatible
    """
    if isinstance(tensors, dict):
        lens = [_first_dim_len(v) for v in tensors.values()]
        n = min(lens) if len(lens) > 0 else 0
        return {k: _slice_first_dim(v, n) for k, v in tensors.items()}, n, lens

    if isinstance(tensors, (list, tuple)):
        lens = [_first_dim_len(v) for v in tensors]
        n = min(lens) if len(lens) > 0 else 0
        out = type(tensors)(_slice_first_dim(v, n) for v in tensors)
        return out, n, lens

    # 单个对象
    n = _first_dim_len(tensors)
    return _slice_first_dim(tensors, n), n, [n]


def data_to_batches(tensors: Union[List, Tuple, Dict], batch_size: int, shuffle_batches=False, seed=None,
                    drop_remainder=False, map_fn=None, prefetch_size=2, num_shards=1, skip=0, shuffle=False):
    """Transform some tensors data into a dataset of mini-batches"""

    # ✅ 核心：强制对齐第0维长度
    tensors, n, lens = _align_tensors_first_dim(tensors)

    # 如果你想定位是谁不一致，可以临时打开：
    # if len(set(lens)) > 1:
    #     print("[data_to_batches] length mismatch:", lens, "-> aligned to", n)

    if isinstance(skip, int) and n > 0 and skip >= n:
        skip = 0

    dataset = tf.data.Dataset.from_tensor_slices(tensors).skip(count=skip)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed, reshuffle_each_iteration=True)

    if num_shards > 1:
        ds = dataset.shard(num_shards, index=0)

        for shard_index in range(1, num_shards):
            shard = dataset.shard(num_shards, index=shard_index)
            ds = ds.concatenate(shard)

        dataset = ds

    if map_fn is not None:
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              deterministic=True)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if shuffle_batches:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed)

    return dataset.prefetch(buffer_size=prefetch_size)


def data_to_batches_1(tensors: Union[List, Tuple, Dict], batch_size: int, shuffle_batches=False, seed=None,
                      drop_remainder=False, map_fn=None, prefetch_size=2, num_shards=1, skip=0, shuffle=False):
    """Transform some tensors data into a dataset of mini-batches"""

    # ✅ 同样做对齐（避免你切换用这个函数时又炸）
    tensors, n, lens = _align_tensors_first_dim(tensors)

    if isinstance(skip, int) and n > 0 and skip >= n:
        skip = 0

    dataset = tf.data.Dataset.from_tensor_slices(tensors).skip(count=skip)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed, reshuffle_each_iteration=True)

    if num_shards > 1:
        ds = dataset.shard(num_shards, index=0)

        for shard_index in range(1, num_shards):
            shard = dataset.shard(num_shards, index=shard_index)
            ds = ds.concatenate(shard)

        dataset = ds

    if map_fn is not None:
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              deterministic=True)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if shuffle_batches:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed)

    return dataset.prefetch(buffer_size=prefetch_size)


def tf_to_scalar_shape(tensor):
    return tf.reshape(tensor, shape=[])


def assert_shapes(a, b):
    assert tf.shape(a) == tf.shape(b)


def tf_01_scaling(x):
    x -= tf.reduce_min(x)
    x /= tf.reduce_max(x)
    return x


def softplus(value=1.0):
    @tf.function
    def activation(x):
        return tf.nn.softplus(x) + value

    return activation


def swish6(x):
    return tf.minimum(tf.nn.swish(x), 6.0)


def dsilu(x):
    """dSiLu activation function (i.e. the derivative of SiLU/Swish).
       Paper: Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning
    """
    sigma_x = tf.nn.sigmoid(x)
    return sigma_x * (1.0 + x * (1.0 - sigma_x))


@tf.function
def batch_norm_relu6(layer: tf.keras.layers.Layer):
    """BatchNormalization + ReLU6, use as activation function"""
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.nn.relu6(layer)
    return layer


@tf.function
def lisht(x):
    """Non-Parameteric Linearly Scaled Hyperbolic Tangent Activation Function"""
    return tf.multiply(x, tf.nn.tanh(x))


@tf.function
def mish(x):
    """A Self Regularized Non-Monotonic Neural Activation Function"""
    return tf.multiply(x, tf.nn.tanh(tf.nn.softplus(x)))


@tf.function
def kl_divergence(log_a, log_b):
    """Kullback-Leibler divergence"""
    return log_a * (log_a - log_b)


@tf.function
def tf_entropy(prob, log_prob):
    return -tf.reduce_sum(prob * log_prob)


def to_int(tensor):
    """Casts the given tensor to tf.int32 datatype"""
    return tf.cast(tensor, dtype=tf.int32)


def to_float(tensor):
    """Casts the given tensor to tf.float32 datatype"""
    return tf.cast(tensor, dtype=tf.float32)


def tf_dot_product(x, y, axis=0, keepdims=False):
    return tf.reduce_sum(tf.multiply(x, y), axis=axis, keepdims=keepdims)


def tf_flatten(x):
    """Reshapes the given input as a 1-D array"""
    return tf.reshape(x, shape=[-1])


# -------------------------------------------------------------------------------------------------
# -- File utils
# -------------------------------------------------------------------------------------------------

def makedir(*args: str) -> str:
    """Creates a directory"""
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path


def file_names(dir_path: str, sort=True) -> list:
    files = filter(lambda f: os.path.isfile(os.path.join(dir_path, f)) and f.startswith('trace-')
                   and f.endswith('.npz'), os.listdir(dir_path))
    if sort:
        files = sorted(files)

    return list(files)


def load_traces(traces_dir: str, max_amount: Optional[int] = None, shuffle=False, offset=0):
    assert offset >= 0

    if shuffle:
        trace_names = file_names(traces_dir, sort=False)
        random.shuffle(trace_names)
    else:
        trace_names = file_names(traces_dir, sort=True)

    if max_amount is None:
        max_amount = np.inf

    for i in range(offset, len(trace_names)):
        name = trace_names[i]
        if i >= max_amount:
            return
        print(f'loading {name}...')
        yield np.load(file=os.path.join(traces_dir, name))


def count_traces(traces_dir: str) -> int:
    """Returns the number of traces available at the given folder."""
    return len(file_names(traces_dir, sort=False))


def unpack_trace(trace: dict, unpack=True) -> Union[tuple, dict]:
    """Reads a trace (i.e. a dict-like object created by np.load()) and unpacks it as a tuple"""
    trace_keys = trace.keys()
    trace = {k: trace[k] for k in trace_keys}  # copy

    for name in ['state', 'action']:
        if sum(k.startswith(name) for k in trace_keys) == 1:
            continue

        keys = filter(lambda k: k.startswith(name + '_'), trace_keys)
        trace[name] = {k: trace[k] for k in keys}

    if 'done' not in trace:
        trace['done'] = None

    if unpack:
        return trace['state'], trace['action'], to_float(trace['reward']), trace['done']

    for key in trace_keys:
        if 'state' in key or 'action' in key:
            if key != 'state' and key != 'action':
                trace.pop(key)

    return trace


def copy_folder(src: str, dst: str):
    """Source: https://stackoverflow.com/a/31039095"""
    dir_util.copy_tree(src, dst)


# -------------------------------------------------------------------------------------------------
# -- Statistics utils
# -------------------------------------------------------------------------------------------------

class Summary:
    def __init__(self, mode='summary', name=None, summary_dir='logs', keys: List[str] = None):
        self.stats = dict()

        if isinstance(keys, list):
            self.allowed_keys = {k: True for k in keys}
        else:
            self.allowed_keys = None

        if mode == 'summary':
            self.should_log = True
            self.use_summary = True

        elif mode == 'log':
            self.should_log = True
            self.use_summary = False
        else:
            self.should_log = False
            self.use_summary = False

        if self.use_summary:
            self.summary_dir = os.path.join(summary_dir, name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.tf_summary_writer = tf.summary.create_file_writer(self.summary_dir)

    def log(self, **kwargs):
        if not self.should_log:
            return

        for key, value in kwargs.items():
            if not self.should_log_key(key):
                continue

            if key not in self.stats:
                self.stats[key] = dict(step=0, list=[])

            # tf tensor: 永远 append（不拆）
            if tf.is_tensor(value):
                self.stats[key]['list'].append(value)
                continue

            # numpy array: 永远 append（不拆）
            if isinstance(value, np.ndarray):
                self.stats[key]['list'].append(value)
                continue

            # list/tuple: 如果是纯标量序列，可以 extend；否则 append
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                if all(np.isscalar(v) for v in value):
                    self.stats[key]['list'].extend(value)
                else:
                    self.stats[key]['list'].append(value)
                continue

            self.stats[key]['list'].append(value)

    def should_log_key(self, key: str) -> bool:
        if self.allowed_keys is None:
            return True
        return key in self.allowed_keys

    def write_summaries(self):
        if not self.use_summary:
            return

        with self.tf_summary_writer.as_default():
            for summary_name, data in self.stats.items():
                step = data['step']
                values = data['list']
                if values is None or len(values) == 0:
                    continue

                imgs = []
                for v in values:
                    try:
                        t = tf.convert_to_tensor(v)
                    except Exception:
                        continue

                    r = t.shape.rank
                    if r is None:
                        try:
                            r = len(t.numpy().shape)
                        except Exception:
                            continue

                    if r == 4:
                        imgs.append(t)
                    elif r == 3:
                        imgs.append(tf.expand_dims(t, 0))
                    elif r == 2:
                        imgs.append(tf.expand_dims(tf.expand_dims(t, 0), -1))
                    else:
                        pass

                if len(imgs) > 0:
                    try:
                        tf.summary.image(summary_name, tf.concat(imgs, axis=0), step=step)
                    except Exception as e:
                        print(f"[Summary] image write failed for {summary_name}: {e}")

                    self.stats[summary_name]['step'] += len(values)
                    self.stats[summary_name]['list'].clear()
                    continue

                flat_list = []
                for v in values:
                    try:
                        t = tf.convert_to_tensor(v, dtype=tf.float32)
                        t = tf.reshape(t, [-1])
                        flat_list.append(t)
                    except Exception:
                        continue

                if len(flat_list) == 0:
                    self.stats[summary_name]['step'] += len(values)
                    self.stats[summary_name]['list'].clear()
                    continue

                x = tf.concat(flat_list, axis=0)

                try:
                    tf.summary.histogram(summary_name, x, step=step)
                except Exception as e:
                    print(f"[Summary] histogram write failed for {summary_name}: {e}")

                try:
                    tf.summary.scalar(summary_name, tf.reduce_mean(x), step=step)
                except Exception as e:
                    print(f"[Summary] scalar write failed for {summary_name}: {e}")

                self.stats[summary_name]['step'] += len(values)
                self.stats[summary_name]['list'].clear()

            self.tf_summary_writer.flush()

    def plot(self, colormap='Set3'):
        num_plots = len(self.stats.keys())
        cmap = plt.get_cmap(name=colormap)
        rows = round(math.sqrt(num_plots))
        cols = math.ceil(math.sqrt(num_plots))

        for k, (key, value) in enumerate(self.stats.items()):
            plt.subplot(rows, cols, k + 1)
            plt.plot(value, color=cmap(k + 1))
            plt.title(key)

        plt.show()


class IncrementalStatistics:
    """Compute mean, variance, and standard deviation incrementally."""
    def __init__(self, epsilon=NP_EPS, max_count=10e8):
        self.mean = 0.0
        self.variance = 0.0
        self.std = 0.0
        self.count = 0

        self.eps = epsilon
        self.max_count = int(max_count)

    def update(self, x, normalize=False):
        x = to_float(x)
        n = tf.shape(x)[0]

        if self.count == 0:
            self.mean = tf.reduce_mean(x)
            self.variance = tf.math.reduce_variance(x) + self.eps
            self.std = tf.sqrt(self.variance)
            self.count = min(int(n.numpy()), self.max_count) if tf.executing_eagerly() else n
            return self.normalize(x) if normalize else None

    def normalize(self, values, eps=NP_EPS):
        return to_float((values - self.mean) / (self.std + eps))

    def set(self, mean: float, variance: float, std: float, count: int):
        self.mean = mean
        self.variance = variance
        self.std = std
        self.count = count

    def as_dict(self) -> dict:
        return dict(mean=np.float(self.mean), variance=np.float(self.variance),
                    std=np.float(self.std), count=np.int(self.count))
