# Reinforcement Learning for Neural Architecture Search

A comprehensive implementation of Neural Architecture Search (NAS) using state-of-the-art reinforcement learning algorithms. This project provides a complete framework for automatically discovering optimal neural network architectures through RL-based optimization.

## Features

- **Modern RL Algorithms**: PPO, SAC, Rainbow DQN, and more
- **Flexible Architecture Search**: Support for various neural network architectures
- **Multiple Environments**: NAS, GridWorld, CartPole for testing
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Visualization Tools**: Interactive dashboards and analysis plots
- **Configuration Management**: YAML/JSON configuration system
- **Checkpointing**: Model saving and loading
- **Unit Tests**: Comprehensive test coverage
- **Type Hints**: Full type annotation support

## 📁 Project Structure

```
├── src/
│   ├── agents/           # RL algorithms and NAS controllers
│   │   ├── nas_controller.py
│   │   └── rl_algorithms.py
│   ├── envs/             # Environment implementations
│   │   └── environments.py
│   ├── utils/            # Utilities and configuration
│   │   ├── config.py
│   │   └── visualization.py
│   └── train.py          # Main training script
├── tests/                # Unit tests
├── configs/              # Configuration files
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Reinforcement-Learning-for-Neural-Architecture-Search.git
   cd Reinforcement-Learning-for-Neural-Architecture-Search
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Training

```bash
# Train with default configuration
python src/train.py

# Train with specific algorithm
python src/train.py --algorithm ppo --total_timesteps 50000

# Train on different environment
python src/train.py --environment gridworld --algorithm sac
```

### Using Configuration Files

```bash
# Create default configuration files
python src/train.py --create_configs

# Train with custom configuration
python src/train.py --config configs/ppo.yaml
```

### Example Configurations

**PPO Configuration** (`configs/ppo.yaml`):
```yaml
algorithm: ppo
environment: nas
total_timesteps: 100000
eval_freq: 1000
save_freq: 5000
log_freq: 100
device: auto
seed: 42
algorithm_config:
  learning_rate: 3e-4
  gamma: 0.99
  batch_size: 64
env_config:
  dataset_name: mnist
  batch_size: 128
  num_epochs: 1
  learning_rate: 0.01
use_wandb: false
use_tensorboard: true
log_dir: ./logs
```

**SAC Configuration** (`configs/sac.yaml`):
```yaml
algorithm: sac
environment: nas
total_timesteps: 100000
algorithm_config:
  learning_rate: 3e-4
  gamma: 0.99
  tau: 0.005
  batch_size: 256
env_config:
  dataset_name: mnist
  batch_size: 128
  num_epochs: 1
```

## Available Algorithms

### 1. Proximal Policy Optimization (PPO)
- **Best for**: Stable policy learning
- **Features**: Clipped objective, value function, entropy bonus
- **Usage**: `--algorithm ppo`

### 2. Soft Actor-Critic (SAC)
- **Best for**: Continuous action spaces
- **Features**: Off-policy learning, entropy maximization
- **Usage**: `--algorithm sac`

### 3. Rainbow DQN
- **Best for**: Discrete action spaces
- **Features**: Distributional RL, multi-step learning
- **Usage**: `--algorithm rainbow_dqn`

## Available Environments

### 1. Neural Architecture Search (NAS)
- **Purpose**: Search for optimal neural network architectures
- **Datasets**: MNIST, CIFAR-10
- **Usage**: `--environment nas`

### 2. GridWorld
- **Purpose**: Navigation task for algorithm testing
- **Features**: Configurable grid size, goal-based rewards
- **Usage**: `--environment gridworld`

### 3. CartPole
- **Purpose**: Classic control task
- **Features**: Continuous state, discrete actions
- **Usage**: `--environment cartpole`

## Visualization and Analysis

### Training Progress
```bash
# Generate visualization report
python src/utils/visualization.py --results logs/final_results.json --output visualizations/
```

### Available Visualizations
- **Training Progress**: Learning curves, loss plots
- **Architecture Analysis**: Performance heatmaps, size distributions
- **Interactive Dashboard**: Plotly-based interactive plots
- **Algorithm Comparison**: Side-by-side performance comparison

### Example Output
The visualization module generates:
- `training_progress.png`: Learning curves and metrics
- `architecture_analysis.png`: Architecture performance analysis
- `interactive_dashboard.html`: Interactive Plotly dashboard
- `architecture_heatmap.png`: Layer-wise performance heatmap
- `training_summary.json`: Comprehensive training summary

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_components.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Monitoring and Logging

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# View in browser
open http://localhost:6006
```

### Weights & Biases
Enable W&B logging in your configuration:
```yaml
use_wandb: true
```

## 🔧 Configuration

### Training Configuration
- `algorithm`: RL algorithm to use
- `environment`: Environment to train on
- `total_timesteps`: Total training steps
- `eval_freq`: Evaluation frequency
- `save_freq`: Checkpoint saving frequency
- `device`: Device to use (cpu/cuda/mps/auto)

### Algorithm Configuration
- `learning_rate`: Learning rate
- `gamma`: Discount factor
- `batch_size`: Training batch size
- `buffer_size`: Replay buffer size (for off-policy algorithms)

### Environment Configuration
- `dataset_name`: Dataset to use (mnist/cifar10)
- `batch_size`: Data loading batch size
- `num_epochs`: Number of training epochs per architecture
- `learning_rate`: Model training learning rate

## Usage Examples

### 1. Basic NAS Training
```python
from src.train import RLNastrainer
from src.utils.config import TrainingConfig

# Create configuration
config = TrainingConfig(
    algorithm="ppo",
    environment="nas",
    total_timesteps=50000,
    env_config={"dataset_name": "mnist"}
)

# Train
trainer = RLNastrainer(config)
results = trainer.train()
```

### 2. Custom Architecture Search
```python
from src.agents.nas_controller import NASController, ArchitectureConfig

# Create custom architecture configuration
arch_config = ArchitectureConfig(
    n_layers=3,
    hidden_size_choices=[64, 128, 256, 512],
    max_layers=5
)

# Create controller
controller = NASController(arch_config)

# Generate architecture
architecture, log_prob, info = controller.forward()
print(f"Generated architecture: {architecture}")
```

### 3. Environment Testing
```python
from src.envs.environments import EnvironmentFactory

# Create environment
env = EnvironmentFactory.create_environment("gridworld", size=5)

# Run episode
state, _ = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action
    state, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}")
```

## Advanced Features

### Checkpointing
```python
from src.utils.config import CheckpointManager

# Save checkpoint
checkpoint_manager = CheckpointManager("./checkpoints")
checkpoint_path = checkpoint_manager.save_checkpoint(
    model, optimizer, step=1000, metrics={"reward": 0.85}
)

# Load checkpoint
checkpoint_manager.load_checkpoint(model, optimizer, checkpoint_path)
```

### Custom Metrics
```python
from src.utils.config import MetricsTracker

# Track custom metrics
tracker = MetricsTracker()
tracker.add_metric("custom_metric", 0.5, step=100)
mean_value = tracker.get_mean("custom_metric")
```

### Early Stopping
```python
from src.utils.config import EarlyStopping

# Setup early stopping
early_stopping = EarlyStopping(patience=10, mode='max')

# Check during training
if early_stopping(eval_reward):
    print("Early stopping triggered!")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Solution: Use `device: cpu` or reduce batch size

2. **Slow Training**
   - Solution: Reduce `num_epochs` in env_config or use smaller datasets

3. **Import Errors**
   - Solution: Ensure you're in the project root and have installed dependencies

4. **Configuration Errors**
   - Solution: Use `--create_configs` to generate default configurations

### Performance Tips

1. **For Faster Training**:
   - Use `num_epochs: 1` in env_config
   - Reduce `total_timesteps`
   - Use smaller datasets

2. **For Better Results**:
   - Increase `total_timesteps`
   - Use more sophisticated algorithms (SAC, Rainbow DQN)
   - Enable logging for monitoring

3. **For Debugging**:
   - Use mock environments (gridworld, cartpole)
   - Enable verbose logging
   - Run unit tests

## References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Rainbow DQN](https://arxiv.org/abs/1710.02298)
- [Neural Architecture Search](https://arxiv.org/abs/1611.01578)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for environment interfaces
- PyTorch for deep learning framework
- Stable Baselines3 for RL algorithm implementations
- The RL research community for algorithm development
 
 
# Reinforcement-Learning-for-Neural-Architecture-Search
