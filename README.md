# SAC-CARLA: Hierarchical Planner Selection with Soft Actor-Critic

A reinforcement learning framework for autonomous driving in CARLA simulator, featuring a SAC-based high-level planner selector that dynamically chooses between Rule-based, Imitation Learning (IL), and Reinforcement Learning (PPO) planners.

## Overview

This project implements a hierarchical decision-making system for autonomous driving:

- **High-Level Agent (SAC)**: Learns to select the most appropriate planner for different driving scenarios
- **Three Low-Level Planners**:
  - **Rule-Based**: Deterministic planner for simple scenarios
  - **IL (Imitation Learning)**: Learned from expert demonstrations
  - **RL (PPO)**: Trained end-to-end for complex maneuvers

The system is trained and tested on multiple scenarios including plain roads, cone obstacles, parked vehicles, and overtaking maneuvers in the CARLA simulator.

## Features

- **Hierarchical Architecture**: SAC agent selects planners based on state observations
- **Multiple Training Scenarios**:
  - Plain: Basic lane following
  - Cones: Static obstacle avoidance
  - Parked Obstacles: Realistic parking scenarios
  - Overtaking: Dynamic multi-vehicle interaction
- **Flexible Configuration**: XML-based route definition and scenario randomization
- **Real-time Visualization**: Pygame-based rendering with HUD display
- **Reward Monitoring**: Comprehensive reward component tracking and visualization

## System Requirements

### Software Dependencies

- Python 3.7+
- CARLA 0.9.15
- PyTorch
- Gym
- NumPy
- Pygame
- OpenCV (optional, for additional visualization)

### Hardware Requirements

- GPU recommended for training (CUDA-compatible)
- Minimum 8GB RAM
- CARLA simulator running on same or networked machine

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd SAC_carla
```

2. **Install CARLA 0.9.15**:
```bash
# Download and extract CARLA 0.9.15
# Update CARLA_ROOT path in carla_base/carla_env.py (line 15)
```

3. **Install Python dependencies**:
```bash
pip install torch torchvision numpy gym pygame
```

4. **Configure paths**:
- Edit `carla_base/carla_env.py` line 15 to set your CARLA installation path
- Update XML waypoint paths in `config.py` if using custom routes

## Quick Start

### 1. Start CARLA Server

```bash
cd /path/to/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20
```

### 2. Train SAC with Frozen Planners

```bash
python train_sac_with_frozen_planners.py
```

### 3. Train Standalone PPO Planner

```bash
python train_ppo_standalone.py
```

### 4. Run Simplified PPO Training

```bash
python train_ppo_simple.py
```

### 5. Test Trained Model

```bash
python main.py --test --model_path weights/sac_model.pth
```

## Project Structure

```
SAC_carla/
├── agent_base/              # SAC agent implementation
│   ├── agent.py            # SAC agent class
│   ├── algorithms.py       # SAC algorithm
│   ├── model.py            # Neural network models
│   └── rule_based_agent.py # Rule-based planner
├── carla_base/             # CARLA environment wrapper
│   ├── carla_env.py        # Main environment
│   ├── carla_sync_mode.py  # Synchronous mode manager
│   └── carla_weather.py    # Weather configuration
├── planners/               # Low-level planners
│   ├── base.py            # Base planner interface
│   ├── il_planner.py      # Imitation learning planner
│   ├── rl_ppo_adapter.py  # PPO adapter
│   └── rl_agent_only/     # Standalone PPO implementation
│       ├── agents/        # PPO agent
│       ├── networks/      # Neural network architectures
│       └── parameters/    # Hyperparameters
├── weights/               # Model checkpoints
├── result/               # Training results
├── config.py            # Configuration parameters
├── main.py              # Main training/testing script
├── train_sac_with_frozen_planners.py  # SAC training
├── train_ppo_standalone.py            # PPO training
├── train_ppo_simple.py               # Simplified PPO
├── router.py            # Planner routing logic
├── replay_memory.py     # Experience replay buffer
├── training_logger.py   # Training metrics logger
└── reward_monitor.py    # Reward component tracking
```

## Configuration

### Main Configuration (`config.py`)

Key parameters:

```python
# Environment
map_name = "Town05"
max_episode_steps = 2000
scenario = "parked_obstacles"  # plain, cones, parked_obstacles

# Agent
obs_dim = 9  # [x, y, z, pitch, yaw, roll, acc, ang_vel, vel]
action_dim = 2  # [throttle/brake, steer]

# Training
num_episodes = 1000
batch_size = 256
gamma = 0.99
tau = 0.005
lr_actor = 3e-4
lr_critic = 3e-4

# Scenarios
cone_num = 10
num_parked_cars = 4
parked_car_spacing = 50.0
```

### Scenario-Specific Parameters

**Cones Scenario**:
- `cone_num`: Number of traffic cones
- `cone_step_behind`: Distance between cones (meters)
- `cone_step_lateral`: Lateral progression per cone

**Parked Obstacles**:
- `num_parked_cars`: Number of parked vehicles
- `parked_car_spacing`: Distance between parked cars
- `parked_car_offset`: Lateral offset from lane center

## Training

### SAC Planner Selector Training

The SAC agent learns to select optimal planners:

```bash
python train_sac_with_frozen_planners.py \
    --episodes 1000 \
    --scenario parked_obstacles \
    --render
```

Training features:
- Frozen low-level planners (pre-trained)
- High-level state observations (9D)
- Discrete planner selection action space
- Multi-component reward shaping

### PPO Planner Training

Train the RL planner independently:

```bash
python train_ppo_standalone.py \
    --total-timesteps 1000000 \
    --scenario cones \
    --render
```

PPO features:
- Continuous action space (throttle/brake, steer)
- On-policy learning with GAE
- Entropy-regularized objectives
- Automatic mixed precision training

## Reward Structure

The reward function combines multiple components:

1. **Waypoint Following** (`r_wp`): Progress toward next waypoint
2. **Speed Maintenance** (`r_speed`): Encouraging optimal velocity
3. **Lane Keeping** (`r_lane`): Penalizing lane deviation
4. **Collision Avoidance** (`r_collision`): Large penalty for crashes
5. **Anti-Idling** (`r_idle`): Preventing standstill behavior

Total reward:
```
R = r_wp + r_speed + r_lane + r_collision + r_idle + base_reward
```

Weights are tunable via `RewardMonitorConfig`.

## Evaluation

### Metrics

- **Success Rate**: Percentage of episodes without collision/timeout
- **Average Speed**: Mean velocity during episode
- **Lane Deviation**: Average distance from lane center
- **Planner Selection**: Distribution of selected planners
- **Collision Rate**: Percentage of collision events

### Visualization

Enable real-time visualization:

```python
config.render = True
config.spectator_mode = "chase"  # or "fixed"
```

## Advanced Usage

### Custom Scenarios

Define custom routes via XML:

```xml
<route town="Town05">
    <waypoints>
        <position x="100.0" y="200.0" z="0.3" yaw="90.0"/>
        <position x="150.0" y="200.0" z="0.3" yaw="90.0"/>
    </waypoints>
</route>
```

Set in config:
```python
config.scenario = "cones_xml"
config.xml_file = "path/to/route.xml"
```

### Multi-Town Training

Enable randomized town selection:

```python
config.randomize_town = True
config.town_pool = ["Town01", "Town03", "Town05"]
```

### Custom Reward Tuning

Override default reward weights:

```python
from reward_monitor import RewardMonitorConfig

custom_reward = RewardMonitorConfig(
    w_speed=2.0,
    w_collision=-10.0,
    w_lane_deviation=-1.5,
    enable_vis=True
)
config.reward_config = custom_reward
```

## Troubleshooting

### Common Issues

1. **CARLA Connection Failed**:
   - Ensure CARLA server is running
   - Check port configuration (default: 2000)
   - Verify firewall settings

2. **Traffic Manager Port Conflict**:
   - The system automatically tries alternative ports
   - Manually specify via `tm_port` parameter
   - Kill existing TM processes: `pkill -f TrafficManager`

3. **GPU Out of Memory**:
   - Reduce batch size in config
   - Enable gradient accumulation
   - Use smaller network architecture

4. **Slow Training**:
   - Disable rendering: `config.render = False`
   - Use `-RenderOffScreen` flag for CARLA
   - Reduce `max_episode_steps`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sac_carla_2024,
  title={SAC-CARLA: Hierarchical Planner Selection for Autonomous Driving},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/SAC_carla}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- CARLA Simulator Team for the autonomous driving platform
- Stable Baselines3 for RL algorithm implementations
- OpenAI for PPO algorithm research

## Contact

For questions and support:
- GitHub Issues: [Create an issue]
- Email: your.email@example.com

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Changelog

### Version 1.0.0 (Current)
- Initial release
- SAC planner selector implementation
- Three-planner hierarchical architecture
- Multiple scenario support
- Comprehensive reward monitoring
- Real-time visualization

## Roadmap

- [ ] Add more planners (e.g., Model Predictive Control)
- [ ] Multi-agent scenarios
- [ ] Transfer learning across towns
- [ ] Integration with CARLA Leaderboard
- [ ] Distributed training support
- [ ] Web-based visualization dashboard
