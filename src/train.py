"""
Main training script for Neural Architecture Search with Reinforcement Learning.

This script provides a complete training pipeline with modern RL algorithms,
visualization, and comprehensive logging.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.agents.nas_controller import NASController, AdvancedNASController, ArchitectureConfig
from src.agents.rl_algorithms import RLAlgorithmFactory, RLConfig
from src.envs.environments import EnvironmentFactory, EnvironmentConfig
from src.utils.config import (
    TrainingConfig, ConfigManager, Logger, CheckpointManager, 
    MetricsTracker, EarlyStopping, set_seed, get_device
)


class RLNastrainer:
    """Main trainer class for RL-based Neural Architecture Search."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_device() if config.device == "auto" else config.device
        
        # Set random seed
        set_seed(config.seed)
        
        # Initialize components
        self._setup_environment()
        self._setup_controller()
        self._setup_algorithm()
        self._setup_logging()
        self._setup_checkpointing()
        
        # Training state
        self.step = 0
        self.episode = 0
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(patience=20, mode='max')
        
    def _setup_environment(self):
        """Setup the training environment."""
        env_config = EnvironmentConfig(**self.config.env_config)
        self.env = EnvironmentFactory.create_environment(
            self.config.environment, config=env_config
        )
        
    def _setup_controller(self):
        """Setup the NAS controller."""
        arch_config = ArchitectureConfig()
        
        if self.config.algorithm in ["sac", "td3"]:
            self.controller = AdvancedNASController(
                arch_config, 
                controller_hidden_size=128,
                device=self.device
            )
        else:
            self.controller = NASController(
                arch_config,
                controller_hidden_size=64,
                device=self.device
            )
    
    def _setup_algorithm(self):
        """Setup the RL algorithm."""
        rl_config = RLConfig(**self.config.algorithm_config)
        rl_config.device = self.device
        
        self.algorithm = RLAlgorithmFactory.create_algorithm(
            self.config.algorithm,
            self.controller,
            rl_config
        )
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = Logger(self.config)
    
    def _setup_checkpointing(self):
        """Setup checkpointing."""
        self.checkpoint_manager = CheckpointManager()
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.logger.info("Starting training...")
        
        # Initialize environment
        state, _ = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        
        episode_rewards = []
        episode_lengths = []
        best_reward = float('-inf')
        
        progress_bar = tqdm(total=self.config.total_timesteps, desc="Training")
        
        while self.step < self.config.total_timesteps:
            episode_reward = 0
            episode_length = 0
            
            # Episode loop
            while True:
                # Select action
                if hasattr(self.algorithm, 'select_action'):
                    action, log_prob, value = self.algorithm.select_action(state)
                else:
                    # Fallback for algorithms without select_action method
                    architecture, log_prob, info = self.controller()
                    action = hash(tuple(architecture)) % 1000
                    value = torch.tensor(0.0)
                
                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = torch.FloatTensor(next_state).to(self.device)
                
                # Store experience
                if hasattr(self.algorithm, 'store_experience'):
                    self.algorithm.store_experience(state, action, reward, log_prob, value, done)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                self.step += 1
                
                # Log metrics
                if self.step % self.config.log_freq == 0:
                    self._log_metrics(info)
                
                # Algorithm update
                if hasattr(self.algorithm, 'update'):
                    update_metrics = self.algorithm.update()
                    if update_metrics:
                        self.logger.log_dict(update_metrics, self.step)
                
                # Evaluation
                if self.step % self.config.eval_freq == 0:
                    eval_reward = self.evaluate()
                    self.logger.log_scalar("eval/reward", eval_reward, self.step)
                    
                    # Early stopping check
                    if self.early_stopping(eval_reward):
                        self.logger.logger.info(f"Early stopping at step {self.step}")
                        break
                
                # Save checkpoint
                if self.step % self.config.save_freq == 0:
                    self._save_checkpoint()
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'reward': f"{episode_reward:.2f}",
                    'best': f"{best_reward:.2f}",
                    'step': self.step
                })
                
                # Episode end
                if done or truncated:
                    break
                
                state = next_state
            
            # Episode finished
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            self.episode += 1
            
            # Update best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # Log episode metrics
            self.logger.log_scalar("episode/reward", episode_reward, self.episode)
            self.logger.log_scalar("episode/length", episode_length, self.episode)
            self.logger.log_scalar("episode/best_reward", best_reward, self.episode)
            
            # Reset environment
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
        
        progress_bar.close()
        
        # Final evaluation and results
        final_results = self._final_evaluation()
        self._save_final_results(final_results)
        
        self.logger.close()
        
        return final_results
    
    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the current policy."""
        total_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            episode_reward = 0
            
            while True:
                with torch.no_grad():
                    architecture, _, _ = self.controller(deterministic=True)
                    action = hash(tuple(architecture)) % 1000
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = torch.FloatTensor(next_state).to(self.device)
                episode_reward += reward
                
                if done or truncated:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def _log_metrics(self, info: Dict[str, Any]):
        """Log training metrics."""
        for key, value in info.items():
            if isinstance(value, (int, float)):
                self.logger.log_scalar(f"info/{key}", value, self.step)
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.controller,
            self.algorithm.optimizer if hasattr(self.algorithm, 'optimizer') else None,
            self.step,
            {"episode": self.episode}
        )
        self.logger.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """Perform final evaluation and generate results."""
        self.logger.logger.info("Performing final evaluation...")
        
        # Evaluate final policy
        final_reward = self.evaluate(num_episodes=10)
        
        # Generate some sample architectures
        sample_architectures = []
        for _ in range(5):
            arch, _, info = self.controller(deterministic=True)
            sample_architectures.append({
                "architecture": arch,
                "info": info
            })
        
        results = {
            "final_reward": final_reward,
            "total_steps": self.step,
            "total_episodes": self.episode,
            "sample_architectures": sample_architectures,
            "config": self.config.__dict__
        }
        
        return results
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results."""
        results_path = Path(self.config.log_dir) / "final_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.logger.info(f"Final results saved: {results_path}")


def create_default_configs():
    """Create default configuration files."""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Default training config
    default_config = TrainingConfig()
    ConfigManager.save_config(default_config, config_dir / "default.yaml")
    
    # PPO config
    ppo_config = TrainingConfig(
        algorithm="ppo",
        algorithm_config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "batch_size": 64
        }
    )
    ConfigManager.save_config(ppo_config, config_dir / "ppo.yaml")
    
    # SAC config
    sac_config = TrainingConfig(
        algorithm="sac",
        algorithm_config={
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256
        }
    )
    ConfigManager.save_config(sac_config, config_dir / "sac.yaml")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RL for Neural Architecture Search")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                      help="Path to configuration file")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "sac", "rainbow_dqn"],
                      help="RL algorithm to use")
    parser.add_argument("--environment", type=str, choices=["nas", "gridworld", "cartpole"],
                      help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"],
                      help="Device to use for training")
    parser.add_argument("--create_configs", action="store_true",
                      help="Create default configuration files")
    
    args = parser.parse_args()
    
    if args.create_configs:
        create_default_configs()
        print("Default configuration files created in configs/")
        return
    
    # Load configuration
    try:
        config = ConfigManager.load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Creating default configs...")
        create_default_configs()
        config = ConfigManager.load_config(args.config)
    
    # Override config with command line arguments
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.environment:
        config.environment = args.environment
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    if args.device:
        config.device = args.device
    
    # Create trainer and start training
    trainer = RLNastrainer(config)
    results = trainer.train()
    
    print(f"Training completed!")
    print(f"Final reward: {results['final_reward']:.4f}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Total episodes: {results['total_episodes']}")


if __name__ == "__main__":
    main()
