"""
Unit tests for the RL NAS project.

This module contains comprehensive unit tests for all major components.
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.nas_controller import NASController, AdvancedNASController, ArchitectureConfig
from src.agents.rl_algorithms import PPOAgent, SACAgent, RainbowDQNAgent, RLConfig
from src.envs.environments import NASEnvironment, MockGridWorld, MockCartPole, EnvironmentConfig
from src.utils.config import TrainingConfig, ConfigManager, Logger, CheckpointManager


class TestArchitectureConfig(unittest.TestCase):
    """Test ArchitectureConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ArchitectureConfig()
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.hidden_size_choices, [32, 64, 128, 256, 512])
        self.assertEqual(config.max_layers, 5)
        self.assertEqual(config.min_layers, 1)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ArchitectureConfig(
            n_layers=3,
            hidden_size_choices=[64, 128, 256],
            max_layers=4,
            min_layers=2
        )
        self.assertEqual(config.n_layers, 3)
        self.assertEqual(config.hidden_size_choices, [64, 128, 256])
        self.assertEqual(config.max_layers, 4)
        self.assertEqual(config.min_layers, 2)


class TestNASController(unittest.TestCase):
    """Test NASController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ArchitectureConfig()
        self.controller = NASController(self.config, device="cpu")
    
    def test_initialization(self):
        """Test controller initialization."""
        self.assertIsInstance(self.controller, NASController)
        self.assertEqual(self.controller.device, "cpu")
        self.assertEqual(self.controller.n_choices, len(self.config.hidden_size_choices))
    
    def test_forward(self):
        """Test forward pass."""
        architecture, log_prob, info = self.controller.forward()
        
        # Check architecture
        self.assertIsInstance(architecture, list)
        self.assertEqual(len(architecture), self.config.n_layers)
        self.assertTrue(all(size in self.config.hidden_size_choices for size in architecture))
        
        # Check log probability
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, ())
        
        # Check info
        self.assertIsInstance(info, dict)
        self.assertIn("actions", info)
        self.assertIn("probabilities", info)
        self.assertIn("entropy", info)
    
    def test_deterministic_forward(self):
        """Test deterministic forward pass."""
        architecture1, _, _ = self.controller.forward(deterministic=True)
        architecture2, _, _ = self.controller.forward(deterministic=True)
        
        # Deterministic should produce same result
        self.assertEqual(architecture1, architecture2)
    
    def test_get_architecture_probability(self):
        """Test architecture probability calculation."""
        architecture = [64, 128]
        prob = self.controller.get_architecture_probability(architecture)
        
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestAdvancedNASController(unittest.TestCase):
    """Test AdvancedNASController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ArchitectureConfig()
        self.controller = AdvancedNASController(self.config, device="cpu")
    
    def test_initialization(self):
        """Test advanced controller initialization."""
        self.assertIsInstance(self.controller, AdvancedNASController)
        self.assertTrue(self.controller.use_attention)
        self.assertIsNotNone(self.controller.attention)
    
    def test_forward_with_attention(self):
        """Test forward pass with attention."""
        architecture, log_prob, info = self.controller.forward()
        
        self.assertIsInstance(architecture, list)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertIsInstance(info, dict)
        self.assertIn("n_layers", info)


class TestRLAlgorithms(unittest.TestCase):
    """Test RL algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ArchitectureConfig()
        self.controller = NASController(self.config, device="cpu")
        self.rl_config = RLConfig(device="cpu")
    
    def test_ppo_agent(self):
        """Test PPO agent."""
        agent = PPOAgent(self.controller, self.rl_config)
        
        # Test action selection
        state = torch.tensor([[0.0]])
        action, log_prob, value = agent.select_action(state)
        
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        
        # Test experience storage
        agent.store_experience(state, action, 0.5, log_prob, value, False)
        self.assertEqual(len(agent.rewards), 1)
        
        # Test update (should not crash)
        agent.update()
    
    def test_sac_agent(self):
        """Test SAC agent."""
        agent = SACAgent(self.controller, self.rl_config)
        
        # Test action selection
        state = torch.tensor([[0.0]])
        action, log_prob = agent.select_action(state)
        
        self.assertIsInstance(action, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        
        # Test experience storage
        next_state = torch.tensor([[0.1]])
        agent.store_experience(state, action, 0.5, next_state, False)
        self.assertEqual(len(agent.replay_buffer), 1)
        
        # Test update (should not crash)
        agent.update()
    
    def test_rainbow_dqn_agent(self):
        """Test Rainbow DQN agent."""
        agent = RainbowDQNAgent(self.controller, self.rl_config)
        
        # Test action selection
        state = torch.tensor([[0.0]])
        action = agent.select_action(state, epsilon=0.0)
        
        self.assertIsInstance(action, int)
        
        # Test experience storage
        next_state = torch.tensor([[0.1]])
        agent.store_experience(state, action, 0.5, next_state, False)
        
        # Test update (should not crash)
        agent.update()


class TestEnvironments(unittest.TestCase):
    """Test environment classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env_config = EnvironmentConfig()
    
    def test_mock_gridworld(self):
        """Test MockGridWorld environment."""
        env = MockGridWorld(size=5)
        
        # Test reset
        state, info = env.reset()
        self.assertEqual(state.shape, (2,))
        self.assertIsInstance(info, dict)
        
        # Test step
        action = 0  # up
        next_state, reward, done, truncated, info = env.step(action)
        
        self.assertEqual(next_state.shape, (2,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_mock_cartpole(self):
        """Test MockCartPole environment."""
        env = MockCartPole()
        
        # Test reset
        state, info = env.reset()
        self.assertEqual(state.shape, (4,))
        self.assertIsInstance(info, dict)
        
        # Test step
        action = 0  # push left
        next_state, reward, done, truncated, info = env.step(action)
        
        self.assertEqual(next_state.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


class TestConfigManager(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_save_load_yaml(self):
        """Test saving and loading YAML config."""
        config = TrainingConfig(algorithm="ppo", total_timesteps=1000)
        config_path = self.temp_path / "test_config.yaml"
        
        # Save config
        ConfigManager.save_config(config, str(config_path))
        self.assertTrue(config_path.exists())
        
        # Load config
        loaded_config = ConfigManager.load_config(str(config_path))
        self.assertEqual(loaded_config.algorithm, "ppo")
        self.assertEqual(loaded_config.total_timesteps, 1000)
    
    def test_config_save_load_json(self):
        """Test saving and loading JSON config."""
        config = TrainingConfig(algorithm="sac", total_timesteps=2000)
        config_path = self.temp_path / "test_config.json"
        
        # Save config
        ConfigManager.save_config(config, str(config_path))
        self.assertTrue(config_path.exists())
        
        # Load config
        loaded_config = ConfigManager.load_config(str(config_path))
        self.assertEqual(loaded_config.algorithm, "sac")
        self.assertEqual(loaded_config.total_timesteps, 2000)


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint management."""
    
    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        checkpoint_manager = CheckpointManager(str(self.temp_path))
        
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, step=100, metrics={"reward": 0.5}
        )
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Load checkpoint
        loaded_checkpoint = checkpoint_manager.load_checkpoint(
            model, optimizer, checkpoint_path
        )
        
        self.assertEqual(loaded_checkpoint["step"], 100)
        self.assertEqual(loaded_checkpoint["metrics"]["reward"], 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_training_step(self):
        """Test end-to-end training step."""
        # Setup
        config = ArchitectureConfig()
        controller = NASController(config, device="cpu")
        rl_config = RLConfig(device="cpu")
        agent = PPOAgent(controller, rl_config)
        
        env = MockGridWorld(size=3)
        
        # Training step
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        agent.store_experience(state, action, reward, log_prob, value, done)
        
        # Should not crash
        agent.update()
        
        self.assertTrue(True)  # If we get here, test passed


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestArchitectureConfig,
        TestNASController,
        TestAdvancedNASController,
        TestRLAlgorithms,
        TestEnvironments,
        TestConfigManager,
        TestCheckpointManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
