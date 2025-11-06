"""
Environment modules for Neural Architecture Search.

This module provides various environments for testing and training RL agents,
including mock environments and real-world NAS tasks.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import random


@dataclass
class EnvironmentConfig:
    """Configuration for environments."""
    dataset_name: str = "mnist"
    batch_size: int = 128
    num_epochs: int = 1
    learning_rate: float = 0.01
    device: str = "cpu"


class NASEnvironment(gym.Env):
    """
    Neural Architecture Search Environment.
    
    This environment allows RL agents to search for optimal neural network architectures
    by training and evaluating different architectures on a dataset.
    """
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Action space: architecture choices
        self.action_space = spaces.Discrete(1000)  # Simplified action space
        
        # Observation space: current state representation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        # Initialize dataset
        self._load_dataset()
        
        # Training state
        self.current_state = np.array([0.0], dtype=np.float32)
        self.episode_rewards = []
        self.best_accuracy = 0.0
        
    def _load_dataset(self):
        """Load the specified dataset."""
        if self.config.dataset_name.lower() == "mnist":
            self._load_mnist()
        elif self.config.dataset_name.lower() == "cifar10":
            self._load_cifar10()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
    
    def _load_mnist(self):
        """Load MNIST dataset."""
        transform = transforms.ToTensor()
        
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        self.val_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=256, shuffle=False
        )
        
        self.input_size = 28 * 28
        self.num_classes = 10
    
    def _load_cifar10(self):
        """Load CIFAR-10 dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        self.val_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=256, shuffle=False
        )
        
        self.input_size = 32 * 32 * 3
        self.num_classes = 10
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_state = np.array([0.0], dtype=np.float32)
        self.episode_rewards = []
        
        return self.current_state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Convert action to architecture (simplified)
        architecture = self._action_to_architecture(action)
        
        # Train and evaluate architecture
        accuracy = self._train_and_evaluate_architecture(architecture)
        
        # Calculate reward
        reward = self._calculate_reward(accuracy)
        
        # Update state
        self.current_state = np.array([accuracy], dtype=np.float32)
        self.episode_rewards.append(reward)
        
        # Check if episode is done
        done = len(self.episode_rewards) >= 10  # Max 10 architectures per episode
        
        # Update best accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        info = {
            "architecture": architecture,
            "accuracy": accuracy,
            "reward": reward,
            "best_accuracy": self.best_accuracy
        }
        
        return self.current_state, reward, done, False, info
    
    def _action_to_architecture(self, action: int) -> List[int]:
        """Convert action to neural network architecture."""
        # Simple mapping: use action to determine layer sizes
        hidden_sizes = [32, 64, 128, 256, 512]
        
        # Use action to select 2-3 layers
        n_layers = 2 + (action % 2)  # 2 or 3 layers
        architecture = []
        
        for i in range(n_layers):
            size_idx = (action + i) % len(hidden_sizes)
            architecture.append(hidden_sizes[size_idx])
        
        return architecture
    
    def _train_and_evaluate_architecture(self, architecture: List[int]) -> float:
        """Train and evaluate a neural network architecture."""
        # Create model
        model = self._create_model(architecture)
        model.to(self.device)
        
        # Training
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(self.config.num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Limit training for speed
                if batch_idx >= 10:
                    break
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Limit evaluation for speed
                if total >= 1000:
                    break
        
        accuracy = correct / total
        return accuracy
    
    def _create_model(self, architecture: List[int]) -> nn.Module:
        """Create neural network model from architecture."""
        layers = [nn.Flatten()]
        
        # Input layer
        layers.append(nn.Linear(self.input_size, architecture[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(architecture[-1], self.num_classes))
        
        return nn.Sequential(*layers)
    
    def _calculate_reward(self, accuracy: float) -> float:
        """Calculate reward based on accuracy."""
        # Simple reward: accuracy with bonus for improvement
        reward = accuracy
        
        # Bonus for beating best accuracy
        if accuracy > self.best_accuracy:
            reward += 0.1
        
        return reward


class MockGridWorld(gym.Env):
    """
    Mock Grid World Environment for testing RL algorithms.
    
    A simple grid world where the agent must navigate to a goal position.
    """
    
    def __init__(self, size: int = 5):
        super().__init__()
        self.size = size
        
        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: agent position
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.int32
        )
        
        # Initialize state
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([size-1, size-1], dtype=np.int32)
        
        # Action mappings
        self.actions = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),   # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])    # right
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        
        return self.agent_pos.copy(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Move agent
        movement = self.actions[action]
        new_pos = self.agent_pos + movement
        
        # Check bounds
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            self.agent_pos = new_pos
        
        # Calculate reward
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -distance_to_goal / (self.size * np.sqrt(2))  # Normalized negative distance
        
        # Check if goal reached
        done = np.array_equal(self.agent_pos, self.goal_pos)
        if done:
            reward += 10.0  # Bonus for reaching goal
        
        info = {
            "agent_pos": self.agent_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "distance_to_goal": distance_to_goal
        }
        
        return self.agent_pos.copy(), reward, done, False, info


class MockCartPole(gym.Env):
    """
    Mock CartPole Environment for testing RL algorithms.
    
    A simplified version of the classic CartPole environment.
    """
    
    def __init__(self):
        super().__init__()
        
        # Action space: push left or right
        self.action_space = spaces.Discrete(2)
        
        # Observation space: cart position, velocity, pole angle, pole velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Environment parameters
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.pole_mass + self.cart_mass
        self.length = 0.5
        self.pole_mass_length = self.pole_mass * self.length
        self.force_magnitude = 10.0
        self.tau = 0.02  # seconds between state updates
        
        # State bounds
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        
        # Initialize state
        self.state = None
        self.steps_beyond_terminated = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Random initial state
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminated = None
        
        return self.state.astype(np.float32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        x, x_dot, theta, theta_dot = self.state
        
        # Apply force
        force = self.force_magnitude if action == 1 else -self.force_magnitude
        
        # Calculate derivatives
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass
        
        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Check termination conditions
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print("Warning: calling step() after episode termination")
            self.steps_beyond_terminated += 1
            reward = 0.0
        
        info = {
            "x": x,
            "x_dot": x_dot,
            "theta": theta,
            "theta_dot": theta_dot
        }
        
        return self.state.astype(np.float32), reward, terminated, False, info


class EnvironmentFactory:
    """Factory class for creating environments."""
    
    @staticmethod
    def create_environment(env_name: str, **kwargs) -> gym.Env:
        """Create environment instance."""
        environments = {
            "nas": NASEnvironment,
            "gridworld": MockGridWorld,
            "cartpole": MockCartPole
        }
        
        if env_name not in environments:
            raise ValueError(f"Unknown environment: {env_name}")
        
        return environments[env_name](**kwargs)
