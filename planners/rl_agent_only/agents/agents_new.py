import os
import gym
import time
import json
import random
import numpy as np
import tensorflow as tf
import logging

from .. import utils
from typing import List, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)


# TODO: actor-critic agent interface (to include policy/value network as well as loading/saving)?
# TODO: save agent configuration as json
class Agent:
    """Agent abstract class"""
    def __init__(self, env: Union[gym.Env, str], batch_size: int, seed=None, weights_dir='weights', name='agent',
                 log_mode='summary', drop_batch_remainder=False, skip_data=0, consider_obs_every=1,
                 evaluation_dir='evaluation', shuffle_batches=False, shuffle=True, traces_dir: str = None,
                 summary_keys: List[str] = None):

        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

        self.seed = None
        self.set_random_seed(seed)

        self.batch_size = batch_size
        self.state_spec = utils.space_to_flat_spec(space=self.env.observation_space, name='state')
        self.action_spec = utils.space_to_flat_spec(space=self.env.action_space, name='action')

        # Record:
        if isinstance(traces_dir, str):
            self.should_record = True
            self.traces_dir = utils.makedir(traces_dir, name)
        else:
            self.should_record = False

        # Data options
        self.drop_batch_remainder = drop_batch_remainder
        self.skip_count = skip_data
        self.obs_skipping = consider_obs_every
        self.shuffle_batches = shuffle_batches
        self.shuffle = shuffle

        # Saving stuff:
        self.base_path = os.path.join(weights_dir, name)
        self.evaluation_path = utils.makedir(os.path.join(evaluation_dir, name))
        self.weights_path = dict(policy=os.path.join(self.base_path, 'policy_net'),
                                 value=os.path.join(self.base_path, 'value_net'))

        # JSON configuration file (keeps track of useful quantities)
        self.config_path = os.path.join(self.base_path, 'config.json')
        self.config = dict()

        # Statistics:
        self.statistics = utils.Summary(mode=log_mode, name=name, keys=summary_keys)

    def set_random_seed(self, seed):
        """Sets the random seed for tensorflow, numpy, python's random, and the environment"""
        if seed is not None:
            assert 0 <= seed < 2**32   
            # assert 0<=seed<1000

            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            self.env.seed(seed)
            self.seed = seed
            print(f'Random seed {seed} set.')

    def act(self, state, *args, **kwargs):
        raise NotImplementedError

    def predict(self, state, *args, **kwargs):
        raise NotImplementedError

    def record(self, *args, **kwargs):
        pass

    def update(self):
        raise NotImplementedError

    def learn(self, *args, **kwargs):
        raise NotImplementedError

    # TODO: re-design `evaluation()` procedure
    def evaluate(self, episodes: int, timesteps: int, render=True, seeds=None) -> list:
        rewards = []
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

            state = self.env.reset()
            state = utils.to_tensor(state)

            # TODO: temporary fix (shouldn't work for deeper nesting...)
            if isinstance(state, dict):
                state = {f'state_{k}': v for k, v in state.items()}

            for t in range(1, timesteps + 1):
                if render:
                    self.env.render()

                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                self.log(actions=action, rewards=reward)

                state = utils.to_tensor(next_state)

                if isinstance(state, dict):
                    state = {f'state_{k}': v for k, v in state.items()}

                if done or (t == timesteps):
                    print(f'Episode {episode} terminated after {t} timesteps with reward {episode_reward}.')
                    rewards.append(episode_reward)
                    break

            self.log(evaluation_reward=episode_reward)
            self.write_summaries()

        self.env.close()

        print(f'Mean reward: {round(np.mean(rewards), 2)}, std: {round(np.std(rewards), 2)}')
        return rewards

    def get_memory(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def test(cls, args: dict, network_summary=False, **kwargs):
        """Rapid testing"""
        agent = cls(**kwargs)

        if network_summary:
            agent.summary()
            breakpoint()

        agent.learn(**args)

    # TODO: one fn for training and another for evaluation?
    def preprocess(self):
        @tf.function
        def preprocess_fn(_):
            return _

        return preprocess_fn

    def log(self, **kwargs):
        self.statistics.log(**kwargs)

    def write_summaries(self):
        try:
            self.statistics.write_summaries()
        except Exception:
            print('[write_summaries] error.')

    def summary(self):
        """Networks summary"""
        raise NotImplementedError

    def update_config(self, **kwargs):
        """Stores the given variables in the configuration dict for later saving"""
        for k, v in kwargs.items():
            self.config[k] = v

    def load_config(self):
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)
            print('config loaded.')
            print(self.config)

    def save_config(self):
        with open(self.config_path, 'w') as file:
            json.dump(self.config, fp=file)
            print('config saved.')

    def reset(self):
        pass

    def load(self):
        """Loads the past agent's state"""
        self.load_weights()    #  由于增加了周围车辆的轨迹预测  权重需要重新训练
        self.load_config()

    def save(self):
        """Saves the agent's state"""
        self.save_weights()
        self.save_config()

    def load_weights(self):
        raise NotImplementedError

    def save_weights(self):
        raise NotImplementedError

    def on_episode_start(self):
        """Episode lifecycle hook that executes at the start of each training episode.
        
        This hook allows agents to:
        1. Reset or initialize episode-specific states
        2. Prepare monitoring metrics
        3. Adjust learning parameters
        4. Configure environment-specific settings
        
        Example uses:
        - Reset episode-specific statistics
        - Update exploration parameters
        - Initialize episode buffers
        - Configure logging for the episode
        - Adjust learning rates or other hyperparameters
        """
        # Reset episode-specific metrics
        self._reset_episode_metrics()
        
        # Update learning parameters if needed
        self._update_learning_parameters()
        
        # Initialize episode monitoring
        self._setup_episode_monitoring()
    
    def _reset_episode_metrics(self) -> None:
        """Reset all episode-specific tracking metrics."""
        self.episode_metrics = {
            'total_reward': 0.0,
            'steps': 0,
            'actions_taken': [],
            'values_predicted': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def _update_learning_parameters(self) -> None:
        """Update any learning parameters that change per episode.
        
        For example:
        - Decay exploration rate
        - Adjust learning rates
        - Update regularization parameters
        """
        if hasattr(self, 'exploration_rate'):
            self.exploration_rate = self._decay_exploration_rate()
            
        if hasattr(self, 'learning_rate'):
            self.learning_rate = self._adjust_learning_rate()
    
    def _setup_episode_monitoring(self) -> None:
        """Configure monitoring and logging for the new episode."""
        # Initialize episode logger
        self.episode_logger = self._create_episode_logger()
        
        # Reset performance monitors
        self.performance_metrics = {
            'fps': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0
        }
        
        logger.debug("Episode monitoring setup completed")
    
    def _decay_exploration_rate(self) -> float:
        """Calculate decayed exploration rate for the new episode.
        
        Returns:
            float: Updated exploration rate
        """
        if not hasattr(self, 'exploration_rate'):
            return 0.0
            
        return max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def _adjust_learning_rate(self) -> float:
        """Adjust learning rate according to schedule or policy.
        
        Returns:
            float: Updated learning rate
        """
        if not hasattr(self, 'learning_rate'):
            return 0.0
            
        return self.learning_rate_scheduler.get_rate()
    
    def _create_episode_logger(self) -> logging.Logger:
        """Create and configure episode-specific logger.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        episode_logger = logging.getLogger(f"{__name__}.episode")
        episode_logger.setLevel(logging.DEBUG)
        return episode_logger

    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing current episode metrics
        """
        return self.episode_metrics.copy()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics.
        
        Returns:
            Dict[str, float]: Dictionary containing performance metrics
        """
        return self.performance_metrics.copy()

    def on_episode_end(self):
        pass


class RandomAgent(Agent):
    def __init__(self, *args, discount=1.0, name='random-agent', repeat_action=1, **kwargs):
        assert 0.0 < discount <= 1.0
        assert repeat_action >= 1

        super().__init__(*args, batch_size=0, name=name, **kwargs)

        self.discount = discount
        self.repeat_action = repeat_action
        self.action_space = self.env.action_space

    def evaluate(self, name: str, timesteps: int, trials: int, render=True, seeds: Union[None, int, List[int]] = None,
                 close=False) -> dict:
        assert trials > 0
        assert timesteps > 0

        if isinstance(seeds, int):
            self.set_random_seed(seed=seeds)

        results = dict(total_reward=[], timesteps=[])
        save_path = os.path.join(self.evaluation_path, f'{name}.json')

        try:
            for trial in range(1, trials + 1):
                # random seed
                if isinstance(seeds, list):
                    if len(seeds) == trials:
                        self.set_random_seed(seed=seeds[trial])
                    else:
                        self.set_random_seed(seed=random.choice(seeds))

                elif seeds == 'sample':
                    self.set_random_seed(seed=random.randint(a=0, b=2 ** 32 - 1))

                self.reset()

                _ = self.env.reset()
                t0 = time.time()
                total_reward = 0.0

                for t in range(timesteps):
                    # Agent prediction
                    action = self.action_space.sample()

                    # Environment step
                    for _ in range(self.repeat_action):
                        next_state, reward, done, _ = self.env.step(action)
                        total_reward += reward

                        if done:
                            break

                    self.log(eval_actions=action, eval_rewards=reward)

                    if done or (t == timesteps):
                        # save results of current trial
                        results['total_reward'].append(total_reward)
                        results['timesteps'].append(t)

                        self.log(**{f'eval_{k}': v[-1] for k, v in results.items()})

                        print(f'Trial-{trial} terminated after {t} timesteps in {round(time.time() - t0, 3)} with total'
                              f' reward of {round(total_reward, 3)}.')
                        break

                self.write_summaries()

            # save average results with their standard deviations over trials as json
            avg_results = {k: np.mean(v) for k, v in results.items()}

            for k, v in results.items():
                avg_results[f'std_{k}'] = np.std(v)

            with open(save_path, 'w') as file:
                json.dump(avg_results, fp=file)

        finally:
            if close:
                self.env.close()

        return results
