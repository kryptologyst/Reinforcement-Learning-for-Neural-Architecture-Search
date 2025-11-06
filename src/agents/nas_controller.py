"""
Neural Architecture Search (NAS) Controller Module.

This module implements a reinforcement learning-based controller for neural architecture search.
The controller generates neural network architectures and receives rewards based on their performance.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass


@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture search."""
    n_layers: int = 2
    hidden_size_choices: List[int] = None
    max_layers: int = 5
    min_layers: int = 1
    
    def __post_init__(self):
        if self.hidden_size_choices is None:
            self.hidden_size_choices = [32, 64, 128, 256, 512]


class NASController(nn.Module):
    """
    Neural Architecture Search Controller using Policy Gradient.
    
    This controller generates neural network architectures by sampling from a policy network.
    It uses reinforcement learning to optimize the architecture generation process.
    """
    
    def __init__(
        self,
        config: ArchitectureConfig,
        controller_hidden_size: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize the NAS Controller.
        
        Args:
            config: Architecture configuration parameters
            controller_hidden_size: Hidden size for the controller network
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__()
        self.config = config
        self.device = device
        self.n_choices = len(config.hidden_size_choices)
        
        # Policy network for generating architectures
        self.policy_net = nn.Sequential(
            nn.Linear(1, controller_hidden_size),
            nn.ReLU(),
            nn.Linear(controller_hidden_size, controller_hidden_size),
            nn.ReLU(),
            nn.Linear(controller_hidden_size, config.n_layers * self.n_choices)
        )
        
        # Layer count predictor (optional extension)
        self.layer_count_predictor = nn.Sequential(
            nn.Linear(1, controller_hidden_size),
            nn.ReLU(),
            nn.Linear(controller_hidden_size, config.max_layers - config.min_layers + 1)
        )
        
        self.to(device)
    
    def forward(self, deterministic: bool = False) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
        """
        Generate a neural network architecture.
        
        Args:
            deterministic: If True, use greedy selection instead of sampling
            
        Returns:
            Tuple containing:
                - architecture: List of hidden layer sizes
                - log_prob: Log probability of the sampled architecture
                - info: Additional information about the generation process
        """
        # Use a dummy input to generate architecture
        dummy_input = torch.tensor([[1.0]], device=self.device)
        
        # Generate logits for each layer
        logits = self.policy_net(dummy_input).view(self.config.n_layers, self.n_choices)
        
        # Create categorical distributions
        distributions = [Categorical(logits=layer_logits) for layer_logits in logits]
        
        if deterministic:
            # Greedy selection
            actions = [dist.probs.argmax() for dist in distributions]
        else:
            # Sample from distributions
            actions = [dist.sample() for dist in distributions]
        
        # Calculate log probabilities
        log_probs = [dist.log_prob(action) for dist, action in zip(distributions, actions)]
        
        # Convert actions to architecture
        architecture = [self.config.hidden_size_choices[action.item()] for action in actions]
        
        # Additional information
        info = {
            "actions": [action.item() for action in actions],
            "probabilities": [dist.probs.detach().cpu().numpy() for dist in distributions],
            "entropy": sum(dist.entropy() for dist in distributions).item()
        }
        
        return architecture, torch.stack(log_probs).sum(), info
    
    def get_architecture_probability(self, architecture: List[int]) -> float:
        """
        Get the probability of generating a specific architecture.
        
        Args:
            architecture: Target architecture to evaluate
            
        Returns:
            Probability of generating the given architecture
        """
        dummy_input = torch.tensor([[1.0]], device=self.device)
        logits = self.policy_net(dummy_input).view(self.config.n_layers, self.n_choices)
        
        total_prob = 1.0
        for i, size in enumerate(architecture):
            if i >= self.config.n_layers:
                break
            choice_idx = self.config.hidden_size_choices.index(size)
            prob = F.softmax(logits[i], dim=0)[choice_idx].item()
            total_prob *= prob
        
        return total_prob
    
    def update_policy(self, rewards: List[float], log_probs: List[torch.Tensor]) -> float:
        """
        Update the policy network using policy gradient.
        
        Args:
            rewards: List of rewards for each architecture
            log_probs: List of log probabilities for each architecture
            
        Returns:
            Policy loss value
        """
        if not rewards or not log_probs:
            return 0.0
        
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        log_probs_tensor = torch.stack(log_probs)
        
        # Normalize rewards (optional)
        if len(rewards) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Calculate policy loss (negative because we want to maximize reward)
        loss = -(log_probs_tensor * rewards_tensor).mean()
        
        return loss


class AdvancedNASController(NASController):
    """
    Advanced NAS Controller with additional features.
    
    Extends the basic controller with:
    - Variable number of layers
    - Attention mechanisms
    - Architecture encoding/decoding
    """
    
    def __init__(
        self,
        config: ArchitectureConfig,
        controller_hidden_size: int = 128,
        use_attention: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize the Advanced NAS Controller.
        
        Args:
            config: Architecture configuration parameters
            controller_hidden_size: Hidden size for the controller network
            use_attention: Whether to use attention mechanism
            device: Device to run the model on
        """
        super().__init__(config, controller_hidden_size, device)
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=controller_hidden_size,
                num_heads=4,
                batch_first=True
            )
    
    def forward(self, deterministic: bool = False) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
        """
        Generate a neural network architecture with advanced features.
        
        Args:
            deterministic: If True, use greedy selection instead of sampling
            
        Returns:
            Tuple containing architecture, log probability, and additional info
        """
        # Generate variable number of layers
        dummy_input = torch.tensor([[1.0]], device=self.device)
        
        # Predict number of layers
        layer_count_logits = self.layer_count_predictor(dummy_input)
        layer_count_dist = Categorical(logits=layer_count_logits)
        
        if deterministic:
            n_layers = layer_count_dist.probs.argmax().item() + self.config.min_layers
        else:
            n_layers = layer_count_dist.sample().item() + self.config.min_layers
        
        # Generate architecture for the determined number of layers
        logits = self.policy_net(dummy_input).view(self.config.n_layers, self.n_choices)
        
        # Use only the first n_layers
        logits = logits[:n_layers]
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention
            logits_reshaped = logits.unsqueeze(0)  # Add batch dimension
            attended_logits, _ = self.attention(logits_reshaped, logits_reshaped, logits_reshaped)
            logits = attended_logits.squeeze(0)
        
        # Create distributions and sample
        distributions = [Categorical(logits=layer_logits) for layer_logits in logits]
        
        if deterministic:
            actions = [dist.probs.argmax() for dist in distributions]
        else:
            actions = [dist.sample() for dist in distributions]
        
        # Calculate log probabilities
        log_probs = [dist.log_prob(action) for dist, action in zip(distributions, actions)]
        
        # Add layer count log probability
        layer_count_log_prob = layer_count_dist.log_prob(torch.tensor(n_layers - self.config.min_layers, device=self.device))
        log_probs.append(layer_count_log_prob)
        
        # Convert to architecture
        architecture = [self.config.hidden_size_choices[action.item()] for action in actions]
        
        # Additional information
        info = {
            "actions": [action.item() for action in actions],
            "n_layers": n_layers,
            "probabilities": [dist.probs.detach().cpu().numpy() for dist in distributions],
            "entropy": sum(dist.entropy() for dist in distributions).item(),
            "layer_count_prob": layer_count_dist.probs.detach().cpu().numpy()
        }
        
        return architecture, torch.stack(log_probs).sum(), info
